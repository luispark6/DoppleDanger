import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

print(parent_dir)
os.chdir(parent_dir)

import asyncio
import json
import numpy as np
import argparse
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, MediaStreamTrack
from aiortc.mediastreams import AudioFrame

import fractions
import queue

import numpy as np
import time
import librosa 
import torch
import torch.nn.functional as F
import sys
import torchaudio
import argparse
from seed_vc.modules.commons import str2bool
from seed_vc.hf_utils import load_custom_model_from_hf
import yaml
from seed_vc.modules.commons import *
import sounddevice as sd
import torchaudio.transforms as tat


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to the model checkpoint")
parser.add_argument("--config-path", type=str, default=None, help="Path to the vocoder checkpoint")
parser.add_argument("--fp16", type=str2bool, nargs="?", const=True, help="Whether to use fp16", default=True)
parser.add_argument("--gpu", type=int, help="Which GPU id to use", default=0)
parser.add_argument('--reference_path', required=True, help='Path to the reference wav file')


args = parser.parse_args()


fp16=True


def load_models(args):
    global fp16
    fp16 = args.fp16
    print(f"Using fp16: {fp16}")
    if args.checkpoint_path is None or args.checkpoint_path == "":
     
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                                         "DiT_uvit_tat_xlsr_ema.pth",
                                                                         "config_dit_mel_seed_uvit_xlsr_tiny.yml")
    else:
        dit_checkpoint_path = args.checkpoint_path
        dit_config_path = args.config_path
    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = 'DiT'
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

    # Load checkpoints
    model, _, _, _ = load_checkpoint(
        model,
        None,
        dit_checkpoint_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Load additional modules
    from seed_vc.modules.campplus.DTDNN import CAMPPlus

    campplus_ckpt_path = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    vocoder_type = model_params.vocoder.type

    if vocoder_type == 'bigvgan':
        from seed_vc.modules.bigvgan import bigvgan
        bigvgan_name = model_params.vocoder.name
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        # remove weight norm in the model and set to eval mode
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)
        vocoder_fn = bigvgan_model
    elif vocoder_type == 'hifigan':
        from seed_vc.modules.hifigan.generator import HiFTGenerator
        from seed_vc.modules.hifigan.f0_predictor import ConvRNNF0Predictor
        hift_config = yaml.safe_load(open('./seed_vc/configs/hifigan.yml', 'r'))
        hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
        hift_path = load_custom_model_from_hf("model-scope/CosyVoice-300M", 'hift.pt', None)
        hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
        hift_gen.eval()
        hift_gen.to(device)
        vocoder_fn = hift_gen
    elif vocoder_type == "vocos":
        vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, 'r'))
        vocos_path = model_params.vocoder.vocos.path
        vocos_model_params = recursive_munch(vocos_config['model_params'])
        vocos = build_model(vocos_model_params, stage='mel_vocos')
        vocos_checkpoint_path = vocos_path
        vocos, _, _, _ = load_checkpoint(vocos, None, vocos_checkpoint_path,
                                         load_only_params=True, ignore_modules=[], is_distributed=False)
        _ = [vocos[key].eval().to(device) for key in vocos]
        _ = [vocos[key].to(device) for key in vocos]
        total_params = sum(sum(p.numel() for p in vocos[key].parameters() if p.requires_grad) for key in vocos.keys())
        print(f"Vocoder model total parameters: {total_params / 1_000_000:.2f}M")
        vocoder_fn = vocos.decoder
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")

    speech_tokenizer_type = model_params.speech_tokenizer.type
    if speech_tokenizer_type == 'whisper':
        # whisper
        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = model_params.speech_tokenizer.name
        whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()],
                                                   return_tensors="pt",
                                                   return_attention_mask=True)
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
            return S_ori
    elif speech_tokenizer_type == 'cnhubert':
        from transformers import (
            Wav2Vec2FeatureExtractor,
            HubertModel,
        )
        hubert_model_name = config['model_params']['speech_tokenizer']['name']
        hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_name)
        hubert_model = HubertModel.from_pretrained(hubert_model_name)
        hubert_model = hubert_model.to(device)
        hubert_model = hubert_model.eval()
        hubert_model = hubert_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy()
                for bib in range(len(waves_16k))
            ]
            ori_inputs = hubert_feature_extractor(ori_waves_16k_input_list,
                                                  return_tensors="pt",
                                                  return_attention_mask=True,
                                                  padding=True,
                                                  sampling_rate=16000).to(device)
            with torch.no_grad():
                ori_outputs = hubert_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    elif speech_tokenizer_type == 'xlsr':
        from transformers import (
            Wav2Vec2FeatureExtractor,
            Wav2Vec2Model,
        )
        model_name = config['model_params']['speech_tokenizer']['name']
        output_layer = config['model_params']['speech_tokenizer']['output_layer']
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
        wav2vec_model = wav2vec_model.to(device)
        wav2vec_model = wav2vec_model.eval()
        wav2vec_model = wav2vec_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy()
                for bib in range(len(waves_16k))
            ]
            ori_inputs = wav2vec_feature_extractor(ori_waves_16k_input_list,
                                                   return_tensors="pt",
                                                   return_attention_mask=True,
                                                   padding=True,
                                                   sampling_rate=16000).to(device)
            with torch.no_grad():
                ori_outputs = wav2vec_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    else:
        raise ValueError(f"Unknown speech tokenizer type: {speech_tokenizer_type}")
    # Generate mel spectrograms
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr,
        "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
        "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    from seed_vc.modules.audio import mel_spectrogram

    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    return (
        model,
        semantic_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    )


def get_device_samplerate():
    return int(
        sd.query_devices(device=sd.default.device[0])["default_samplerate"]
    )


def get_device_channels():
    max_input_channels = sd.query_devices(device=sd.default.device[0])[
        "max_input_channels"
    ]
    max_output_channels = sd.query_devices(device=sd.default.device[1])[
        "max_output_channels"
    ]
    return min(max_input_channels, max_output_channels, 2)

prompt_condition, mel2, style2 = None, None, None
reference_wav_name = ""

prompt_len = 3  # in seconds
ce_dit_difference = 2.0  # 2 seconds

@torch.no_grad()
def custom_infer(model_set,
                 reference_wav,
                 new_reference_wav_name,
                 input_wav_res,
                 block_frame_16k,
                 skip_head,
                 skip_tail,
                 return_length,
                 diffusion_steps,
                 inference_cfg_rate,
                 max_prompt_length,
                 cd_difference=2.0,
                 ):
    global prompt_condition, mel2, style2
    global reference_wav_name
    global prompt_len
    global ce_dit_difference
    (
        model,
        semantic_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    ) = model_set
    sr = mel_fn_args["sampling_rate"]
    hop_length = mel_fn_args["hop_size"]
    if ce_dit_difference != cd_difference:
        ce_dit_difference = cd_difference
        print(f"Setting ce_dit_difference to {cd_difference} seconds.")
    if prompt_condition is None or reference_wav_name != new_reference_wav_name or prompt_len != max_prompt_length:
        prompt_len = max_prompt_length
        print(f"Setting max prompt length to {max_prompt_length} seconds.")
        reference_wav = reference_wav[:int(sr * prompt_len)]
        reference_wav_tensor = torch.from_numpy(reference_wav).to(device)

        ori_waves_16k = torchaudio.functional.resample(reference_wav_tensor, sr, 16000)
        S_ori = semantic_fn(ori_waves_16k.unsqueeze(0))
        feat2 = torchaudio.compliance.kaldi.fbank(
            ori_waves_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = campplus_model(feat2.unsqueeze(0))

        mel2 = to_mel(reference_wav_tensor.unsqueeze(0))
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)
        prompt_condition = model.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
        )[0]

        reference_wav_name = new_reference_wav_name

    converted_waves_16k = input_wav_res
    if device.type == "mps":
        start_event = torch.mps.event.Event(enable_timing=True)
        end_event = torch.mps.event.Event(enable_timing=True)
        torch.mps.synchronize()
    else:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

    start_event.record()
    S_alt = semantic_fn(converted_waves_16k.unsqueeze(0))
    end_event.record()
    if device.type == "mps":
        torch.mps.synchronize()  # MPS - Wait for the events to be recorded!
    else:
        torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Time taken for semantic_fn: {elapsed_time_ms}ms")

    ce_dit_frame_difference = int(ce_dit_difference * 50)
    S_alt = S_alt[:, ce_dit_frame_difference:]
    target_lengths = torch.LongTensor([(skip_head + return_length + skip_tail - ce_dit_frame_difference) / 50 * sr // hop_length]).to(S_alt.device)
    print(f"target_lengths: {target_lengths}")
    cond = model.length_regulator(
        S_alt, ylens=target_lengths , n_quantizers=3, f0=None
    )[0]
    cat_condition = torch.cat([prompt_condition, cond], dim=1)
    with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
        vc_target = model.cfm.inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
            mel2,
            style2,
            None,
            n_timesteps=diffusion_steps,
            inference_cfg_rate=inference_cfg_rate,
        )
        vc_target = vc_target[:, :, mel2.size(-1) :]
        print(f"vc_target.shape: {vc_target.shape}")
        vc_wave = vocoder_fn(vc_target).squeeze()
    output_len = return_length * sr // 50
    tail_len = skip_tail * sr // 50
    output = vc_wave[-output_len - tail_len: -tail_len]

    return output


model_set = load_models(args)
SAMPLERATE= 48000
print(f"Sample rate: {SAMPLERATE}")
BLOCK_TIME = 0.3
VAD_CHUNK_SIZE = 1000 * BLOCK_TIME
VAD_SPEECH_DETECTED = False
SET_SPEECH_DETECTED_FALSE_AT_END_FLAG = False
EXTRA_TIME_CE = 2.0
ZC = SAMPLERATE // 50  
EXTRA_FRAME = (
    int(
        np.round(
            EXTRA_TIME_CE
            * SAMPLERATE
            / ZC
        )
    )
    * ZC
)

CROSSFADE_TIME = 0.1
CROSSFADE_TIME = (
    int(
        np.round(
            CROSSFADE_TIME
            * SAMPLERATE
            / ZC
        )
    )
    * ZC
)
SOLA_SEARCH_FRAME = ZC

BLOCK_FRAME = (
    int(
        np.round(
            BLOCK_TIME
            * SAMPLERATE
            / ZC
        )
    )
    * ZC
)
print(f"BLOCK_FRAME: {BLOCK_FRAME}")

EXTRA_TIME_RIGHT = 0.02
EXTRA_FRAME_RIGHT = (
    int(
        np.round(
            EXTRA_TIME_RIGHT
            * SAMPLERATE
            / ZC
        )
    )
    * ZC
)
INPUT_WAV: torch.Tensor = torch.zeros(
    EXTRA_FRAME
    + CROSSFADE_TIME
    + SOLA_SEARCH_FRAME
    + BLOCK_FRAME
    + EXTRA_FRAME_RIGHT,
    device=device,
    dtype=torch.float32,
) 
INPUT_WAV_RES: torch.Tensor = torch.zeros(
    320 * INPUT_WAV.shape[0] // ZC,
    device=device,
    dtype=torch.float32,
)
BLOCK_FRAME_16K = 320 * BLOCK_FRAME // ZC
EXTRA_TIME  = 0.5
REFERECE_AUDIO_PATH = args.reference_path
REFERENCE_WAV, _ = librosa.load(
    REFERECE_AUDIO_PATH, sr=model_set[-1]["sampling_rate"]
)
SKIP_HEAD = EXTRA_FRAME // ZC
SKIP_TAIL = EXTRA_FRAME_RIGHT // ZC
CROSSFADE_FRAME = (
    int(
        np.round(
            CROSSFADE_TIME
            * SAMPLERATE
            / ZC
        )
    )
    * ZC
)

SOLA_BUFFER_FRAME = min(CROSSFADE_FRAME, 4 * ZC)
DIFFUSION_STEPS = 8
prev_time = time.time()

RETURN_LENGTH = (
    BLOCK_FRAME + SOLA_BUFFER_FRAME + SOLA_SEARCH_FRAME
) // ZC

INFERENCE_CFG_RATE = 0.7
MAX_PROMPT_LENGTH = 3.0 

SOLA_BUFFER: torch.Tensor = torch.zeros(
    SOLA_BUFFER_FRAME, device=device, dtype=torch.float32
)

if model_set[-1]["sampling_rate"] != SAMPLERATE:
    RESAMPLER2 = tat.Resample(
        orig_freq=model_set[-1]["sampling_rate"],
        new_freq=SAMPLERATE,
        dtype=torch.float32,
    ).to(device)
else: RESAMPLER2 = None

FADE_IN_WINDOW: torch.Tensor = (
    torch.sin(
        0.5
        * np.pi
        * torch.linspace(
            0.0,
            1.0,
            steps=SOLA_BUFFER_FRAME,
            device=device,
            dtype=torch.float32,
        )
    )
    ** 2
)
FADE_OUT_WINDOW: torch.Tensor = 1 - FADE_IN_WINDOW
CHANNELS = 1
print(f"CHANNELS: {CHANNELS}")


from funasr import AutoModel
vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")

def audio_callback(
        indata: np.ndarray, outdata: np.ndarray, frames, times, status
    ):
        """
        Audio block callback function
        """
        print("entering audio call back", file=sys.stderr, flush=True)
        global VAD_SPEECH_DETECTED
        global SET_SPEECH_DETECTED_FALSE_AT_END_FLAG
        
        start_time = time.perf_counter()
        indata = librosa.to_mono(indata.T)

        # VAD first
        if device.type == "mps":
            start_event = torch.mps.event.Event(enable_timing=True)
            end_event = torch.mps.event.Event(enable_timing=True)
            torch.mps.synchronize()
        else:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
        start_event.record()
        indata_16k = librosa.resample(indata, orig_sr=SAMPLERATE, target_sr=16000)
        print(f"INDATA: {indata.shape}", file=sys.stderr, flush=True)
        res = vad_model.generate(input=indata_16k, cache={}, is_final=False, chunk_size=VAD_CHUNK_SIZE)
        res_value = res[0]["value"]
        print(f"res value: {res_value}")
        print(f"speech detected: {VAD_SPEECH_DETECTED}")

        if len(res_value) % 2 == 1 and not VAD_SPEECH_DETECTED:
            VAD_SPEECH_DETECTED = True
        elif len(res_value) % 2 == 1 and VAD_SPEECH_DETECTED:
            SET_SPEECH_DETECTED_FALSE_AT_END_FLAG = True

        if len(res_value) % 2 == 0 and VAD_SPEECH_DETECTED:
            VAD_SPEECH_DETECTED = False


        end_event.record()
        if device.type == "mps":
            torch.mps.synchronize()  # MPS - Wait for the events to be recorded!
        else:
            torch.cuda.synchronize()  # Wait for the events to be recorded!
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Time taken for VAD: {elapsed_time_ms}ms")


        INPUT_WAV[: -BLOCK_FRAME] = INPUT_WAV[
            BLOCK_FRAME :
        ].clone()
        INPUT_WAV[-indata.shape[0] :] = torch.from_numpy(indata).to(
           device
        )
        INPUT_WAV_RES[: -BLOCK_FRAME_16K] = INPUT_WAV_RES[
            BLOCK_FRAME_16K :
        ].clone()
        INPUT_WAV_RES[-320 * (indata.shape[0] // ZC + 1) :] = (
            # self.resampler(self.input_wav[-indata.shape[0] - 2 * self.zc :])[
            #     320:
            # ]
            torch.from_numpy(librosa.resample(INPUT_WAV[-indata.shape[0] - 2 * ZC :].cpu().numpy(), orig_sr=SAMPLERATE, target_sr=16000)[320:])
        )
        print(f"preprocess time: {time.perf_counter() - start_time:.2f}")
        # infer
     
        if EXTRA_TIME_CE - EXTRA_TIME < 0:
            raise ValueError("Content encoder extra context must be greater than DiT extra context!")
        if device.type == "mps":
            start_event = torch.mps.event.Event(enable_timing=True)
            end_event = torch.mps.event.Event(enable_timing=True)
            torch.mps.synchronize()
        else:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
        start_event.record()
        infer_wav = custom_infer(
            model_set,
            REFERENCE_WAV,
            REFERECE_AUDIO_PATH,
            INPUT_WAV_RES,
            BLOCK_FRAME_16K,
            SKIP_HEAD,
            SKIP_TAIL,
            RETURN_LENGTH,
            int(DIFFUSION_STEPS),
            INFERENCE_CFG_RATE,
            MAX_PROMPT_LENGTH,
            EXTRA_TIME_CE - EXTRA_TIME,
        )
        if RESAMPLER2 is not None:
            infer_wav = RESAMPLER2(infer_wav)
        end_event.record()
        if device.type == "mps":
            torch.mps.synchronize()  # MPS - Wait for the events to be recorded!
        else:
            torch.cuda.synchronize()  # Wait for the events to be recorded!
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Time taken for VC: {elapsed_time_ms}ms")
        if not VAD_SPEECH_DETECTED:
            print("---------------------------------------------------------------------------------------")
            print(res_value)
            infer_wav = torch.zeros_like(INPUT_WAV[EXTRA_FRAME :])
            
    
        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
        conv_input = infer_wav[
            None, None, : SOLA_BUFFER_FRAME + SOLA_SEARCH_FRAME
        ]

        cor_nom = F.conv1d(conv_input, SOLA_BUFFER[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, SOLA_BUFFER_FRAME, device=device),
            )
            + 1e-8
        )

        tensor = cor_nom[0, 0] / cor_den[0, 0]
        if tensor.numel() > 1:  # If tensor has multiple elements
            if sys.platform == "darwin":
                _, sola_offset = torch.max(tensor, dim=0)
                sola_offset = sola_offset.item()
            else:
                sola_offset = torch.argmax(tensor, dim=0).item()
        else:
            sola_offset = tensor.item()

        print(f"sola_offset = {int(sola_offset)}")

        #post_process_start = time.perf_counter()
        infer_wav = infer_wav[sola_offset:]
        infer_wav[: SOLA_BUFFER_FRAME] *= FADE_IN_WINDOW
        infer_wav[: SOLA_BUFFER_FRAME] += (
            SOLA_BUFFER * FADE_OUT_WINDOW
        )
        SOLA_BUFFER[:] = infer_wav[
            BLOCK_FRAME : BLOCK_FRAME + SOLA_BUFFER_FRAME
        ]
        outdata[:] = (
            infer_wav[: BLOCK_FRAME]
            .repeat(CHANNELS, 1)
            .t()
            .cpu()
            .numpy()
        )

        total_time = time.perf_counter() - start_time
    
        if SET_SPEECH_DETECTED_FALSE_AT_END_FLAG:
            VAD_SPEECH_DETECTED = False
            SET_SPEECH_DETECTED_FALSE_AT_END_FLAG = False
        print(f"Infer time: {total_time:.2f}")

        return outdata






def process_audio(chunk: bytes) -> bytes:
    # Example: amplify the volume
    int16_audio =np.frombuffer(chunk, dtype=np.int16)
    audio = int16_audio.astype(np.float32)/32767
    samples = audio.shape[0]
 
    outdata = np.zeros((samples, 1), dtype=np.float32)
    processed = audio_callback(audio, outdata, None, None, None)
    processed_int = (processed*32767).astype(np.int16)

    return processed_int.tobytes()

async def run_peer_b():
    print("Waiting for offer.json from Peer A...")
    while not os.path.exists("./webrtc_windows/offer.json"):
        await asyncio.sleep(1)
        print("Still waiting...")

    print("Found offer.json!")

    config = RTCConfiguration([
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
    ])
    pc = RTCPeerConnection(configuration=config)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "disconnected"):
            print("Connection lost.")

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print(f"ICE connection state: {pc.iceConnectionState}")

    
    @pc.on("datachannel")
    def on_datachannel(channel):
        print(f"🔌 Received DataChannel: {channel.label}")

        @channel.on("message")
        def on_message(message):
            global prev_time
            if isinstance(message, bytes):
                print(f"📥 Received {len(message)} bytes")

                processed = process_audio(message)
                channel.send(processed)
                print(f"📤 Sent back {len(processed)} bytes")
                cur_time = time.time()
                dur = cur_time - prev_time
                print(f"Time: {dur}")
                prev_time = cur_time

            else:
                print(f"⚠️ Received non-bytes message: {message}")

    # Load the offer
    with open("./webrtc_windows/offer.json", "r") as f:
        offer_data = json.load(f)

    offer = RTCSessionDescription(
        sdp=offer_data["sdp"],
        type=offer_data["type"]
    )
    await pc.setRemoteDescription(offer)
    print("Remote description (offer) set")

    # Create and send the answer
    print("Creating answer...")
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    print("Waiting for ICE gathering to complete...")
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.1)

    answer_data = {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }

    with open("./webrtc_windows/answer.json", "w") as f:
        json.dump(answer_data, f, indent=2)

    print("\n✅ STEP 2 complete: Answer and ICE candidates created!")
    print("Now return to sender and press Enter to continue the connection.")

    try:
        while True:
            await asyncio.sleep(5)
            print(f"Status - Connection: {pc.connectionState}, ICE: {pc.iceConnectionState}")
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        await pc.close()


if __name__ == "__main__":
    asyncio.run(run_peer_b())