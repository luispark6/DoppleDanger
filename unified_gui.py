import os
import sys
from dotenv import load_dotenv
load_dotenv()

os.environ["OMP_NUM_THREADS"] = "4"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)
import warnings
import yaml

warnings.simplefilter("ignore")

from seed_vc.modules.commons import *
import librosa
import torchaudio

from seed_vc.hf_utils import load_custom_model_from_hf

import os
import sys
import torch
from seed_vc.modules.commons import str2bool
import subprocess
import swap_live_video


# Load model and configuration
device = None

flag_vc = False

prompt_condition, mel2, style2 = None, None, None
reference_wav_name = ""

prompt_len = 3  # in seconds
ce_dit_difference = 2.0  # 2 seconds
fp16 = False
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

def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)

class Config:
    def __init__(self):
        self.device = device


if __name__ == "__main__":
    import re
    import threading
    import time
    from multiprocessing import cpu_count
    import argparse
    import librosa
    import numpy as np
    import sounddevice as sd
    import torch
    import torch.nn.functional as F
    import torchaudio.transforms as tat
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QSlider, QComboBox, QCheckBox,  QLineEdit,
        QFileDialog, QMessageBox, QFrame, QMessageBox, QListView, QGraphicsDropShadowEffect, QTabWidget, QGridLayout
    )
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor, QIntValidator
    from functools import partial

    


    current_dir = os.getcwd()
    n_cpu = cpu_count()

    
    class ClickableSlider(QSlider):
        def __init__(self, orientation, parent=None):
            super().__init__(orientation, parent)
            self._dragging = False

            self.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 2px;
                background: #1e1e1e;  /* Darker groove */
                border-radius: 1px;
            }

            QSlider::handle:horizontal {
                background: #c084fc;
                border: none;
                width: 10px;
                height: 10px;
                margin: -4px 0;
                border-radius: 5px;
            }

            QSlider::handle:horizontal:hover {
                background: #a855f7;
            }

            QSlider::sub-page:horizontal {
                background: #a855f7;
                border-radius: 1px;
            }

            QSlider::add-page:horizontal {
                background: #1e1e1e;  /* Match groove for consistent dark look */
                border-radius: 1px;
            }
        """)

        def mousePressEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton:
                self._dragging = True
                self.updateSliderValue(event)
                event.accept()
            else:
                super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            if self._dragging:
                self.updateSliderValue(event)
                event.accept()
            else:
                super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton:
                self._dragging = False
                event.accept()
            else:
                super().mouseReleaseEvent(event)

        def updateSliderValue(self, event):
            pos = event.position()
            ratio = float(pos.x()) / self.width() if self.orientation() == Qt.Orientation.Horizontal else float(pos.y()) / self.height()
            ratio = max(0.0, min(1.0, ratio))
            if self.orientation() == Qt.Orientation.Vertical:
                ratio = 1.0 - ratio
            value_range = self.maximum() - self.minimum()
            new_val = round(self.minimum() + ratio * value_range)
            self.setValue(new_val)

    class ModernComboBox(QComboBox): 
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setView(QListView())
            self.setStyleSheet("""
            QComboBox {
                border: 1px solid #c084fc;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 13px;
                background-color: #3a3a3a;
                color: #E0E0E0;
            }

            QComboBox:focus, QComboBox:pressed {
                background-color: #3a3a3a;
                color: #E0E0E0;
                border: 1px solid #d8b4fe;
            }

            QComboBox:hover {
                border: 1px solid #d8b4fe;
                background-color: #a855f7;
            }

            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 12px;
                border-left: 1px solid #c084fc;
                background: transparent;
            }

            QComboBox::down-arrow {
                image: none;
                width: 0;
                height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #c084fc;
                margin-top: 2px;
            }

            QComboBox QAbstractItemView {
                border: 1px solid #c084fc;
                border-radius: 6px;
                background-color: #3a3a3a;
                color: #E0E0E0;
                padding: 0px;
                margin: 0px;
                outline: 0px;
            }

            QComboBox QAbstractItemView::item {
                padding: 6px 10px;
                background-color: #3a3a3a;
                color: #E0E0E0;
            }

            QComboBox QAbstractItemView::item:hover {
                background-color: #4a4a4a;
            }

            QComboBox QAbstractItemView::item:selected {
                background-color: #a855f7;
                color: white;
            }
                                    
        """)
        
            # Create shadow effect
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(5)  # Adjust blur for glow intensity
            shadow.setColor(QColor(192, 132, 252, 100))  # Purple with transparency
            shadow.setOffset(0, 0)  # Center the glow
            
            # Apply the effect
            self.setGraphicsEffect(shadow)


    class ModernButton(QPushButton):
        def __init__(self, text="", parent=None):
            super().__init__(text, parent)
            self.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                color: #E0E0E0;
                border: 1px solid #c084fc;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 13px;
            }
            
            QPushButton:hover {
                background-color: #a855f7;
                border: 1px solid #d8b4fe;
            }
            
            QPushButton:pressed {
                background-color: #3a3a3a;
                color: #E0E0E0;
                border: 1px solid #d8b4fe;
            }
            
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #808080;
                border: 1px solid #555555;
            }
            """)
            # Create glowing purple effect
            glow = QGraphicsDropShadowEffect()
            glow.setBlurRadius(5)
            glow.setOffset(0, 0)  # No offset for glow effect
            glow.setColor(QColor(192, 132, 252, 120))  # Purple glow with transparency
            self.setGraphicsEffect(glow)
    class ModernCheckBox(QCheckBox):
        def __init__(self, text="", parent=None):
            super().__init__(text, parent)
            
            # Setup initial glow effect
            self.setup_glow_effect()
            
            # Apply the modern purple color scheme
            self.setStyleSheet("""
                QCheckBox {
                color: #E0E0E0;
                font-size: 13px;
                spacing: 8px;
                background-color: transparent;
            }

            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #c084fc;
                border-radius: 3px;
                background-color: #3a3a3a;
            }

            QCheckBox::indicator:hover {
                border: 1px solid #d8b4fe;
                background-color: #4a4a4a;
            }

            QCheckBox::indicator:pressed {
                border: 1px solid #d8b4fe;
                background-color: #a855f7;
            }

            QCheckBox::indicator:checked {
                border: 1px solid #d8b4fe;
                background-color: #a855f7;
                image: none;
            }

            QCheckBox::indicator:checked:hover {
                border: 1px solid #d8b4fe;
                background-color: #9333ea;
            }

            QCheckBox::indicator:checked:pressed {
                border: 1px solid #d8b4fe;
                background-color: #7c3aed;
            }

            QCheckBox:hover {
                color: #f0f0f0;
            }

            QCheckBox:disabled {
                color: #666666;
            }

            QCheckBox::indicator:disabled {
                border: 1px solid #666666;
                background-color: #2a2a2a;
            }
                            
                        """)
        
        def setup_glow_effect(self):
            """Setup the base glow effect"""
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(5)
            shadow.setColor(QColor(192, 132, 252, 80))  # Purple with transparency
            shadow.setOffset(0, 0)  # Center the glow
            self.setGraphicsEffect(shadow)
        
        def enterEvent(self, event):
            """Enhanced glow on hover"""
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(5)
            shadow.setColor(QColor(216, 180, 254, 120))  # Brighter purple glow
            shadow.setOffset(0, 0)
            self.setGraphicsEffect(shadow)
            super().enterEvent(event)
        
        def leaveEvent(self, event):
            """Return to normal glow"""
            self.setup_glow_effect()
            super().leaveEvent(event)


    class GUIConfig:
        def __init__(self) -> None:
            self.reference_audio_path: str = ""
            # self.index_path: str = ""
            self.diffusion_steps: int = 10
            self.sr_type: str = "sr_model"
            self.block_time: float = 0.25  # s
            self.threhold: int = -60
            self.crossfade_time: float = 0.05
            self.extra_time_ce: float = 2.5
            self.extra_time: float = 0.5
            self.extra_time_right: float = 2.0
            self.I_noise_reduce: bool = False
            self.O_noise_reduce: bool = False
            self.inference_cfg_rate: float = 0.7
            self.sg_hostapi: str = ""
            self.wasapi_exclusive: bool = False
            self.sg_input_device: str = ""
            self.sg_output_device: str = ""
            self.sg_wasapi_exclusive = False
            self.max_prompt_length: float = 3
    class MainWindow(QMainWindow):
        def __init__(self, args):
            
            super().__init__()

            self.setWindowTitle("DoppleDanger")
            self.setMinimumSize(950, 400)
            self.resize(950, 600)  # Set a default size larger than minimum


            # Apply dark stylesheet
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #1e1e1e;
                }

                QWidget {
                    background-color: #2b2b2b;
                    color: #E0E0E0;
                    font-size: 14px;
                    font-family: "Segoe UI", sans-serif;
                }

                QFrame {
                    background-color: #2b2b2b;
                }

                QFrame[frameShape="4"] {  /* QFrame.Shape.VLine = 4 */
                    border-left: 2px solid #444;
                    margin: 0 10px;
                }
                
                QTabWidget::pane {
                    border: 1px solid #444;
                    background-color: #2b2b2b;
                }
                
                QTabBar::tab {
                    background-color: #3c3c3c;
                    color: #E0E0E0;
                    padding: 8px 16px;
                    margin-right: 2px;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }
                
                QTabBar::tab:selected {
                    background-color: #2b2b2b;
                    border-bottom: 2px solid #0078d4;
                }
                
                QTabBar::tab:hover {
                    background-color: #404040;
                }
            """)

            # Central widget and tab widget
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QVBoxLayout()
            central_widget.setLayout(main_layout)
            
            # Create tab widget
            self.tab_widget = QTabWidget()
            main_layout.addWidget(self.tab_widget)
            
            # Create first tab (Main Application)
            self.create_main_tab()
            
            # Create second tab (Guide)
            self.create_guide_tab()
            
            # Initialize your existing variables
            self.gui_config = GUIConfig()
            self.config = Config()
            self.function = "vc"
            self.delay_time = 0
            self.hostapis = None
            self.input_devices = None
            self.output_devices = None
            self.input_devices_indices = None
            self.output_devices_indices = None
            self.stream = None
            self.process = None
            self.model_set = load_models(args)


            from funasr import AutoModel
            self.vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")

            self.update_devices()  # hostapi, input/output devices, and indices now exist

            self.voice_clone_gui()
            self.face_clone_gui()

        def create_main_tab(self):
            """Create the main application tab with your existing UI"""
            main_tab = QWidget()
            main_tab_layout = QHBoxLayout()
            main_tab.setLayout(main_tab_layout)
            
            # Left Section: Existing UI
            left_widget = QWidget()
            left_layout = QVBoxLayout()
            left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            left_widget.setLayout(left_layout)
            main_tab_layout.addWidget(left_widget, 1)

            # Vertical line as divider
            line = QFrame()
            line.setFrameShape(QFrame.Shape.VLine)
            line.setFrameShadow(QFrame.Shadow.Sunken)
            line.setLineWidth(2)
            main_tab_layout.addWidget(line)

            # Right Section: Placeholder
            right_widget = QWidget()
            right_layout = QVBoxLayout()
            right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            right_widget.setLayout(right_layout)
            main_tab_layout.addWidget(right_widget, 1)

            # Save references (these will be used by your existing methods)
            self.layout = left_layout     
            self.right_layout = right_layout 
            
            # Add the tab to the tab widget
            self.tab_widget.addTab(main_tab, "Main Application")

        def create_guide_tab(self):
            """Create the guide tab"""
            guide_tab = QWidget()
            guide_layout = QVBoxLayout()
            guide_tab.setLayout(guide_layout)
            
            # Add guide content
            from PyQt6.QtWidgets import QLabel, QScrollArea, QTextEdit
            
            # Create a scrollable area for the guide
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            
            # Create guide content widget
            guide_content = QWidget()
            guide_content_layout = QVBoxLayout()
            guide_content.setLayout(guide_content_layout)
            
            # Add guide title
            title_label = QLabel("DoppleDanger User Guide")
            title_label.setStyleSheet("""
                QLabel {
                    font-size: 20px;
                    font-weight: bold;
                    text-decoration: underline;
                    padding: 6px 0;
                }
            """)
            guide_content_layout.addWidget(title_label)
            
            # Add guide content using QTextEdit for rich text
            guide_text = QTextEdit()
            guide_text.setReadOnly(True)
            guide_text.setHtml("""
            <h3>Getting Started</h3>
            <p>Welcome to DoppleDanger! This application provides live face and voice cloning capabilities.</p>
            
            <h3>Voice Cloning Key Components</h3>
            <ul>
                <li>You must import a 5-20 second reference audio clip of the voice you would like to clone. Import the audio clip using the <i>Choose reference Audio</i> button</li>
                <li>Properly choose the input and output device</li>
                <li>For faster and smoother inference, please keep the <i>Diffusion Steps</i> between 4-15. This however depends on the hardware in use. </li>
                <li>Make sure <i>Block Time</i> * 1000 (ms conversion) is greater than the <i>inference time</i> (can only check inference time by running the voice). This is very important! </li>
                <li>Run <i>Start Voice Conversion</i> when your ready!</li>
                <li>For more details on the other parameters, please refer to the repo</li>
            </ul>
            
            <h3>Face Swapping Key Components</h3>
            <ul>
                <li>You must import an image of the person's face you want to clone. Import the image using the <i>Open Source Image</i></li>
                <li>You must import the ReSwapper model that should be in your ./models folder. Import the model using <i>Open Model File</i></li>
                <li>Press <i>Start Face Swap</i> to begin face facewap</li>
                <li>For more details on the other parameters, please refer to the repo</li>
            </ul>
            
            <h3>Tips and Troubleshooting</h3>
            <ul>
                <li>Ensure your microphone is properly connected</li>
                <li>Check audio device permissions</li>
                <li>For best results, use high-quality input audio and input image</li>
                <li>Make sure your system meets the minimum requirements(found in the repo)</li>
            </ul>
        
            """)
            
            guide_text.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    text-decoration: underline;
                    padding: 6px 0;
                }
            """)
            
            guide_content_layout.addWidget(guide_text)
            
            # Set the guide content to the scroll area
            scroll_area.setWidget(guide_content)
            guide_layout.addWidget(scroll_area)
            
            # Add the guide tab to the tab widget
            self.tab_widget.addTab(guide_tab, "User Guide")

        def closeEvent(self, event):
            # This is called when user clicks the X button
            self.stop_stream() 
            self.stop_face_swap()
            event.accept()  # Allow closing

        #row builder
        def add_row(self, *widgets, layout):
            row = QHBoxLayout()
            for widget in widgets:
                row.addWidget(widget)
            layout.addLayout(row)          
        def add_slider_pair(self, label1, slider1, label2, slider2, layout):
            row = QHBoxLayout()
            col1 = QVBoxLayout()
            col1.addWidget(label1)
            col1.addWidget(slider1)

            col2 = QVBoxLayout()
            col2.addWidget(label2)
            col2.addWidget(slider2)

            row.addLayout(col1)
            row.addLayout(col2)
            layout.addLayout(row)

        #face cloning functions (face_clone_gui()->openmodel_file())
        def face_clone_gui(self):
            section_title = QLabel("Face Swapping")
            section_title.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    text-decoration: underline;
                    padding: 6px 0;
                }
            """)
            self.right_layout.addWidget(section_title)

            self.source_face_label = QLabel("Choose Source Face Image*")
            self.source_face_button = ModernButton("Open Source Image")
            self.source_face_button.clicked.connect(self.open_image_file)
            self.add_row(self.source_face_label, self.source_face_button, layout=self.right_layout)

            self.model_label = QLabel("Choose Model File*")
            self.model_button = ModernButton("Open Model File")
            self.model_button.clicked.connect(self.open_model_file)
            self.add_row(self.model_label, self.model_button, layout=self.right_layout)


            self.resolution_label = QLabel("Face Resolution(px)")
            self.resolution_textbox = QLineEdit()
            self.resolution_textbox.setValidator(QIntValidator())
            self.resolution_textbox.setMaximumWidth(70)
            self.resolution_textbox.setText("128")
            self.resolution_textbox.textChanged.connect(self.res_changed)

            self.delay_label = QLabel("Delay(ms)")
            self.delay_textbox = QLineEdit()
            self.delay_textbox.setValidator(QIntValidator())
            self.delay_textbox.setMaximumWidth(70)
            self.delay_textbox.setText("0")
            self.delay_textbox.textChanged.connect(self.delay_changed)

            self.right_layout.setSpacing(15)  # Adds space between widgets in the layout

            
            self.add_row(self.resolution_label, self.resolution_textbox, self.delay_label, self.delay_textbox, layout=self.right_layout)

            self.right_layout.setSpacing(15)

            self.obs_label = QLabel("Enable OBS Virtual Camera")
            self.obs_checkbox = ModernCheckBox()
        

            self.mouth_label = QLabel("Retain Target Mouth              ")
            self.mouth_checkbox = ModernCheckBox()
            self.add_row(self.obs_label, self.obs_checkbox, self.mouth_label, self.mouth_checkbox, layout=self.right_layout)

            self.fps_delay_label = QLabel("Show FPS and Delay            ")
            self.fps_delay_checkbox = ModernCheckBox()

            self.enhance_label = QLabel("Enhance Camera Resolution")
            self.enhance_checkbox = ModernCheckBox()
            self.add_row(self.fps_delay_label, self.fps_delay_checkbox , self.enhance_label, self.enhance_checkbox , layout=self.right_layout)


            self.start_faceswap_button = ModernButton("Start Face Swap")
            self.start_faceswap_button.clicked.connect(self.start_face_swap)
            self.stop_faceswap_button = ModernButton("Stop Face Swap")
            self.stop_faceswap_button.setEnabled(False)
            self.stop_faceswap_button.clicked.connect(self.stop_face_swap)
            self.add_row(self.start_faceswap_button, self.stop_faceswap_button, layout=self.right_layout)
        def monitor_process(self):
            def read_stream(stream, label):
                for line in iter(stream.readline, ''):
                    print(f"[{label}] {line.strip()}")

            threading.Thread(target=read_stream, args=(self.process.stdout, "STDOUT"), daemon=True).start()
            threading.Thread(target=read_stream, args=(self.process.stderr, "STDERR"), daemon=True).start()

            self.process.wait()
        def build_command(self):
            # Assuming your original script is named 'face_swap.py'
            # Define the script name (with or without .py/.exe extension)
            script_name = "swap_live_video"  # Base name (no extension)

            # Check if the current script is running as an EXE (PyInstaller)
            is_frozen = getattr(sys, 'frozen', False)

            # Determine the correct executable/script path
            if is_frozen:
                # Running as EXE → Use .exe version of the script
                script_path = os.path.join(os.path.dirname(sys.executable), f"{script_name}.exe")
            else:
                # Running as .py → Use .py version of the script
                script_path = f"{script_name}.py"

            # Build the command
            cmd = [sys.executable if not is_frozen else script_path]  # Use sys.executable for .py, script_path for .exe
            if not is_frozen:
                cmd.extend([script_path])

            cmd.extend(["--source", self.source_face_button.text()])
            cmd.extend(["--modelPath", self.model_button.text()])
            cmd.extend(["--resolution", self.resolution_textbox.text()])
            cmd.extend(["--delay", self.delay_textbox.text()])
            
            # if self.face_attr_direction.get():
            #     cmd.extend(["--face_attribute_direction", self.face_attr_direction.get()])
            #     cmd.extend(["--face_attribute_steps", str(self.face_attr_steps.get())])
            
            if self.obs_checkbox.isChecked():
                cmd.append("--obs")
            
            if self.mouth_checkbox.isChecked():
                cmd.append("--mouth_mask")
            
            if self.fps_delay_checkbox.isChecked():
                cmd.append("--fps_delay")

            if self.enhance_checkbox.isChecked():
                cmd.append("--enhance_res")
            
            return cmd
        def start_face_swap(self):
            if self.source_face_button.text()=="Open Source Image" or self.model_button.text()=="Open Model File":
                QMessageBox.warning(
                    self,
                    "Input Error",
                    f"Please include a Source Image and a Model File to Start Face Swap"
                )
                return
            
            try:
                cmd = self.build_command()
                self.process = subprocess.Popen(cmd, 
                                            stdout=subprocess.PIPE, 
                                            stderr=subprocess.PIPE,
                                            text=True)
                
                self.start_faceswap_button.setEnabled(False)  # Disables the button
                self.stop_faceswap_button.setEnabled(True)
                print("Face Swap is Running........")
                
                # Start monitoring thread
                threading.Thread(target=self.monitor_process, daemon=True).start()
                
            except Exception as e:
                print(e)
        def stop_face_swap(self):
            if self.process:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                except Exception as e:
                    print(f"Error stopping process: {e}")
            self.process=None
            self.stop_faceswap_button.setEnabled(False)
            self.start_faceswap_button.setEnabled(True)            
        def delay_changed(self, text):
            if not text:
                return  # skip empty input
            if "-" in text:
                QMessageBox.warning(
                    self,
                    "Input Error",
                    f"Value cannot be less than {0}."
                )
                self.delay_textbox.setText(str(0))  
        def res_changed(self, text):
            if not text:
                return  # skip empty input
        
            if "-" in text:
                QMessageBox.warning(
                    self,
                    "Input Error",
                    f"Value cannot be less than {64}."
                )
                self.resolution_textbox.setText(str(64))  
                return

            value = int(text)
            if value > 256:
                QMessageBox.warning(
                    self,
                    "Input Error",
                    f"Value cannot be greater than {256}."
                )
                self.resolution_textbox.setText(str(256))  
            elif value<64:
                QMessageBox.warning(
                    self,
                    "Input Error",
                    f"Value cannot be less than {64}."
                )
                self.resolution_textbox.setText(str(64))  # reset to max
        def open_image_file(self):
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select an Image File",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)"
            )

            if file_path:
                self.source_face_button.setText(f"{file_path}")
        def open_model_file(self):
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select an Model File",
                "",
                "Model Files (*.pt *.pth)"
            )

            if file_path:
                self.model_button.setText(f"{file_path}")

        #voice cloning functions (voice_clone_gui()->update_devices())
        def voice_clone_gui(self):
            self.config = Config()
            section_title = QLabel("Voice Cloning")
            section_title.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    text-decoration: underline;
                    padding: 6px 0;
                }
            """)
            self.layout.addWidget(section_title)

            # Reference audio
            self.ref_audio_label = QLabel("Choose Reference Audio*")
            self.ref_audio_button = ModernButton("Open Audio File")
            self.ref_audio_button.clicked.connect(self.open_audio_file)
            self.add_row(self.ref_audio_label, self.ref_audio_button, layout=self.layout)

            # Host API
            self.hostapi_label = QLabel("Select a Host API:")
            self.hostapi_dropdown = ModernComboBox()
            self.hostapi_dropdown.addItems(self.hostapis)
            self.hostapi_dropdown.view().parentWidget().setStyleSheet('background-color: #3a3a3a;')

            self.hostapi_dropdown.currentIndexChanged.connect(self.hostapi_changed)
            self.add_row(self.hostapi_label, self.hostapi_dropdown, layout=self.layout)

            # WASAPI
            self.wasapi_label = QLabel("WASAPI")
            self.wasapi_checkbox = ModernCheckBox()
            self.add_row(self.wasapi_label, self.wasapi_checkbox, layout=self.layout)

            # Input Device
            self.inputDevice_label = QLabel("Select Input Device:")
            self.inputDevice_dropdown = ModernComboBox()
            self.inputDevice_dropdown.addItems(self.input_devices)
            self.inputDevice_dropdown.view().parentWidget().setStyleSheet('background-color: #3a3a3a;')
            self.add_row(self.inputDevice_label, self.inputDevice_dropdown,layout=self.layout)

            # Output Device
            self.outputDevice_label = QLabel("Select Output Device:")
            self.outputDevice_dropdown = ModernComboBox()
            self.outputDevice_dropdown.addItems(self.output_devices)
            self.outputDevice_dropdown.view().parentWidget().setStyleSheet('background-color: #3a3a3a;')
            self.add_row(self.outputDevice_label, self.outputDevice_dropdown, layout=self.layout)

            # Reload button
            self.refresh_button = ModernButton("Reload Devices")
            self.refresh_button.clicked.connect(self.hostapi_changed)
            self.layout.addWidget(self.refresh_button)

            self.sr_flag = False

            # SR Model
            self.sr_model_label = QLabel("Use Model SR")
            self.sr_model_checkbox = ModernCheckBox()
            self.sr_model_checkbox.stateChanged.connect(partial(self.sr, 1))
            self.sr_model_checkbox.setChecked(True)
            self.add_row(self.sr_model_label, self.sr_model_checkbox, layout=self.layout)

            # SR Device
            self.sr_device_label = QLabel("Use Device SR")
            self.sr_device_checkbox = ModernCheckBox()
            self.sr_device_checkbox.stateChanged.connect(partial(self.sr, 2))
            self.add_row(self.sr_device_label, self.sr_device_checkbox, layout=self.layout)

            self.sr_flag = True

            #diffusion steps
            self.diffusion_label = QLabel("Diffusion Steps: 7")
            self.layout.addWidget(self.diffusion_label)
            self.diffusion_slider = ClickableSlider(Qt.Orientation.Horizontal)
            self.diffusion_slider.setMinimum(1)
            self.diffusion_slider.setMaximum(30)
            self.diffusion_slider.setValue(7)
            self.diffusion_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.diffusion_slider.setTickInterval(1)
            self.diffusion_slider.valueChanged.connect(self.diffusion_changed)


            #Inference CFG Rate Divide by 10
            self.infer_rate_label = QLabel("Inference CFG Rate: 0.7")
            self.layout.addWidget(self.infer_rate_label)
            self.infer_slider = ClickableSlider(Qt.Orientation.Horizontal)
            self.infer_slider.setMinimum(0)
            self.infer_slider.setMaximum(10)
            self.infer_slider.setValue(7)
            self.infer_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.infer_slider.setTickInterval(1)
            self.infer_slider.valueChanged.connect(self.infer_changed)


            self.add_slider_pair(self.diffusion_label, self.diffusion_slider, self.infer_rate_label, self.infer_slider, self.layout)


            #Max Prompt Length divide by 2
            self.maxPrompt_label = QLabel("Max Prompt Length: 3.0")
            self.layout.addWidget(self.maxPrompt_label)

            self.prompt_slider = ClickableSlider(Qt.Orientation.Horizontal)
            self.prompt_slider.setMinimum(2)
            self.prompt_slider.setMaximum(40)
            self.prompt_slider.setValue(6)
            self.prompt_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.prompt_slider.setTickInterval(1)
            self.prompt_slider.valueChanged.connect(self.prompt_length_changed)

            #Block Time divide by 50
            self.blockTime_label = QLabel("Block Time: 0.6")
            self.layout.addWidget(self.blockTime_label)
            self.blockTime_slider = ClickableSlider(Qt.Orientation.Horizontal)
            self.blockTime_slider.setMinimum(2)
            self.blockTime_slider.setMaximum(150)
            self.blockTime_slider.setValue(30)
            self.blockTime_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.blockTime_slider.setTickInterval(1)
            self.blockTime_slider.valueChanged.connect(self.blockTime_changed)
            

            self.add_slider_pair(self.maxPrompt_label, self.prompt_slider, self.blockTime_label, self.blockTime_slider, self.layout)


            #Crossfade length 
            self.crossfade_label = QLabel("Crossfade Length: 0.1")
            self.layout.addWidget(self.crossfade_label)
            self.crossfade_slider = ClickableSlider(Qt.Orientation.Horizontal)
            self.crossfade_slider.setMinimum(1)
            self.crossfade_slider.setMaximum(25)
            self.crossfade_slider.setValue(5)
            self.crossfade_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.crossfade_slider.setTickInterval(1)
            self.crossfade_slider.valueChanged.connect(self.crossfade_slider_changed)
            self.layout.addWidget(self.crossfade_slider)

            #Extra CE Context (left)
            self.extra_ce_L = QLabel("Extra CE Context (left): 2.0")
            self.layout.addWidget(self.extra_ce_L)
            self.extra_ce_L_slider = ClickableSlider(Qt.Orientation.Horizontal)
            self.extra_ce_L_slider.setMinimum(5)
            self.extra_ce_L_slider.setMaximum(100)
            self.extra_ce_L_slider.setValue(20)
            self.extra_ce_L_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.extra_ce_L_slider.setTickInterval(1)
            self.extra_ce_L_slider.valueChanged.connect(self.extra_ce_L_changed)
            self.layout.addWidget(self.extra_ce_L_slider)


            self.add_slider_pair(self.crossfade_label, self.crossfade_slider, self.extra_ce_L, self.extra_ce_L_slider, self.layout)



            #Extra dit Context (left)
            self.extra_DIT_L = QLabel("Extra DiT Context (left): 0.5")
            self.layout.addWidget(self.extra_DIT_L)
            self.extra_DIT_slider = ClickableSlider(Qt.Orientation.Horizontal)
            self.extra_DIT_slider.setMinimum(5)
            self.extra_DIT_slider.setMaximum(100)
            self.extra_DIT_slider.setValue(5)
            self.extra_DIT_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.extra_DIT_slider.setTickInterval(1)
            self.extra_DIT_slider.valueChanged.connect(self.extra_DIT_L_changed)
            self.layout.addWidget(self.extra_DIT_slider)


            #Extra CE Context (Right)
            self.extra_ce_R = QLabel("Extra CE Context (right): 0.02")
            self.layout.addWidget(self.extra_ce_R)
            self.extra_ce_R_slider = ClickableSlider(Qt.Orientation.Horizontal)
            self.extra_ce_R_slider.setMinimum(1)
            self.extra_ce_R_slider.setMaximum(500)
            self.extra_ce_R_slider.setValue(1)
            self.extra_ce_R_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.extra_ce_R_slider.setTickInterval(1)
            self.extra_ce_R_slider.valueChanged.connect(self.extra_ce_R_changed)
            self.layout.addWidget(self.extra_ce_R_slider)

            self.add_slider_pair(self.extra_DIT_L, self.extra_DIT_slider, self.extra_ce_R, self.extra_ce_R_slider, self.layout)

    
           
            # Start/Stop buttons
            self.start_vc_button = ModernButton("Start Voice Conversion")
            self.start_vc_button.clicked.connect(self.init_vc)
            self.stop_vc_button = ModernButton("Stop Voice Conversion")
            self.stop_vc_button.setEnabled(False)
            self.stop_vc_button.clicked.connect(self.stop_stream)
            self.add_row(self.start_vc_button, self.stop_vc_button, layout=self.layout)

            # Inference time display
            self.infer_time_label = QLabel("Inference Time: N/A")
            self.layout.addWidget(self.infer_time_label)
        def set_values(self):
            if self.ref_audio_button.text() == "Open Audio File":
                QMessageBox.information(self, "Missing File", "Choose an audio file.")
                return False
            pattern = re.compile("[^\x00-\x7F]+")
            if pattern.findall(self.ref_audio_button.text()):
                QMessageBox.information(self, "Invalid File Path", "Audio file path contains non-ascii characters.")
                return False
            self.set_devices(self.inputDevice_dropdown.currentText(), self.outputDevice_dropdown.currentText())
            self.gui_config.sg_hostapi = self.hostapi_dropdown.currentText()
            self.gui_config.sg_wasapi_exclusive = self.wasapi_checkbox.isChecked()
            self.gui_config.sg_input_device = self.inputDevice_dropdown.currentText()
            self.gui_config.sg_output_device = self.outputDevice_dropdown.currentText()
            self.gui_config.reference_audio_path = self.ref_audio_button.text()
            self.gui_config.sr_type = ["sr_model", "sr_device"][
                [
                    self.sr_model_checkbox.isChecked(),
                    self.sr_device_checkbox.isChecked(),
                ].index(True)
            ]
            # # self.gui_config.threhold = values["threhold"]
            self.gui_config.diffusion_steps = self.diffusion_slider.value()
            self.gui_config.inference_cfg_rate = self.infer_slider.value()/10
            
            self.gui_config.max_prompt_length = self.prompt_slider.value()/2
            self.gui_config.block_time = self.blockTime_slider.value()/50
            self.gui_config.crossfade_time = self.crossfade_slider.value()/50
            self.gui_config.extra_time_ce = self.extra_ce_L_slider.value()/10
            self.gui_config.extra_time = self.extra_DIT_slider.value()/10
            self.gui_config.extra_time_right = self.extra_ce_R_slider.value()/50
            return True
        def init_vc(self):
            global flag_vc
            if self.set_values()==True and not flag_vc:
                printt("cuda_is_available: %s", torch.cuda.is_available())
                self.start_vc()
        def set_devices(self, input_device, output_device):
            """set input and output devices."""
            sd.default.device[0] = self.input_devices_indices[
                self.input_devices.index(input_device)
            ]
            sd.default.device[1] = self.output_devices_indices[
                self.output_devices.index(output_device)
            ]
            printt("Input device: %s:%s", str(sd.default.device[0]), input_device)
            printt("Output device: %s:%s", str(sd.default.device[1]), output_device)   
        def stop_stream(self):
            global flag_vc
            if flag_vc:
                flag_vc = False
                if self.stream is not None:
                    self.stream.abort()
                    self.stream.close()
                    self.stream = None
                self.stop_vc_button.setEnabled(False)
                self.start_vc_button.setEnabled(True)
        def extra_ce_R_changed(self, value):
            self.extra_ce_R.setText(f"Extra CE Context (right): {value/50}")
        def extra_DIT_L_changed(self,value):
            self.extra_DIT_L.setText(f"Extra DiT Context (left): {value/10}")
        def extra_ce_L_changed(self, value):
            self.extra_ce_L.setText(f"Extra CE Context (left): {value/10}")
        def crossfade_slider_changed(self, value):
            self.crossfade_label.setText(f"Crossfade Length: {value/50}")
        def blockTime_changed(self, value):
            self.blockTime_label.setText(f"Block Time: {value/50}")
        def prompt_length_changed(self, value):
            self.maxPrompt_label.setText(f"Max Prompt Length: {value/2}")
        def infer_changed(self, value):
            self.infer_rate_label.setText(f"Inference CFG Rate: {value/10}")
        def diffusion_changed(self, value):
            self.diffusion_label.setText(f"Diffusion Steps: {value}")
        def start_vc(self):
            if device.type == "mps":
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()
            self.reference_wav, _ = librosa.load(
                self.gui_config.reference_audio_path, sr=self.model_set[-1]["sampling_rate"]
            )
            self.gui_config.samplerate = (
                self.model_set[-1]["sampling_rate"]
                if self.gui_config.sr_type == "sr_model"
                else self.get_device_samplerate()
            )
            self.gui_config.channels = self.get_device_channels()
            self.zc = self.gui_config.samplerate // 50  # 44100 // 100 = 441
            self.block_frame = (
                int(
                    np.round(
                        self.gui_config.block_time
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
            )
            self.block_frame_16k = 320 * self.block_frame // self.zc
            self.crossfade_frame = (
                int(
                    np.round(
                        self.gui_config.crossfade_time
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
            )
            self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
            self.sola_search_frame = self.zc
            self.extra_frame = (
                int(
                    np.round(
                        self.gui_config.extra_time_ce
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
            )
            self.extra_frame_right = (
                    int(
                        np.round(
                            self.gui_config.extra_time_right
                            * self.gui_config.samplerate
                            / self.zc
                        )
                    )
                    * self.zc
            )
            self.input_wav: torch.Tensor = torch.zeros(
                self.extra_frame
                + self.crossfade_frame
                + self.sola_search_frame
                + self.block_frame
                + self.extra_frame_right,
                device=self.config.device,
                dtype=torch.float32,
            )  # 2 * 44100 + 0.08 * 44100 + 0.01 * 44100 + 0.25 * 44100
            self.input_wav_denoise: torch.Tensor = self.input_wav.clone()
            self.input_wav_res: torch.Tensor = torch.zeros(
                320 * self.input_wav.shape[0] // self.zc,
                device=self.config.device,
                dtype=torch.float32,
            )  # input wave 44100 -> 16000
            self.rms_buffer: np.ndarray = np.zeros(4 * self.zc, dtype="float32")
            self.sola_buffer: torch.Tensor = torch.zeros(
                self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
            )
            self.nr_buffer: torch.Tensor = self.sola_buffer.clone()
            self.output_buffer: torch.Tensor = self.input_wav.clone()
            self.skip_head = self.extra_frame // self.zc
            self.skip_tail = self.extra_frame_right // self.zc
            self.return_length = (
                self.block_frame + self.sola_buffer_frame + self.sola_search_frame
            ) // self.zc
            self.fade_in_window: torch.Tensor = (
                torch.sin(
                    0.5
                    * np.pi
                    * torch.linspace(
                        0.0,
                        1.0,
                        steps=self.sola_buffer_frame,
                        device=self.config.device,
                        dtype=torch.float32,
                    )
                )
                ** 2
            )
            self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
            self.resampler = tat.Resample(
                orig_freq=self.gui_config.samplerate,
                new_freq=16000,
                dtype=torch.float32,
            ).to(self.config.device)
            if self.model_set[-1]["sampling_rate"] != self.gui_config.samplerate:
                self.resampler2 = tat.Resample(
                    orig_freq=self.model_set[-1]["sampling_rate"],
                    new_freq=self.gui_config.samplerate,
                    dtype=torch.float32,
                ).to(self.config.device)
            else:
                self.resampler2 = None
            self.vad_cache = {}
            self.vad_chunk_size = 1000 * self.gui_config.block_time
            self.vad_speech_detected = False
            self.set_speech_detected_false_at_end_flag = False
            self.start_stream()
        def start_stream(self):
            global flag_vc
            if not flag_vc:
                flag_vc = True
                if (
                    "WASAPI" in self.gui_config.sg_hostapi
                    and self.gui_config.sg_wasapi_exclusive
                ):
                    extra_settings = sd.WasapiSettings(exclusive=True)
                else:
                    extra_settings = None
                self.stream = sd.Stream(
                    callback=self.audio_callback,
                    blocksize=self.block_frame,
                    samplerate=self.gui_config.samplerate,
                    channels=self.gui_config.channels,
                    dtype="float32",
                    extra_settings=extra_settings,
                )
                self.start_vc_button.setEnabled(False)
                self.stop_vc_button.setEnabled(True)
                self.stream.start()
        def set_devices(self, input_device, output_device):
            """set input and output devices."""
            sd.default.device[0] = self.input_devices_indices[
                self.input_devices.index(input_device)
            ]
            sd.default.device[1] = self.output_devices_indices[
                self.output_devices.index(output_device)
            ]
            printt("Input device: %s:%s", str(sd.default.device[0]), input_device)
            printt("Output device: %s:%s", str(sd.default.device[1]), output_device)
        def get_device_samplerate(self):
            return int(
                sd.query_devices(device=sd.default.device[0])["default_samplerate"]
            )
        def get_device_channels(self):
            max_input_channels = sd.query_devices(device=sd.default.device[0])[
                "max_input_channels"
            ]
            max_output_channels = sd.query_devices(device=sd.default.device[1])[
                "max_output_channels"
            ]
            return min(max_input_channels, max_output_channels, 2)
        def audio_callback(
            self, indata: np.ndarray, outdata: np.ndarray, frames, times, status
        ):
            """
            Audio block callback function
            """
            global flag_vc
            
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
            indata_16k = librosa.resample(indata, orig_sr=self.gui_config.samplerate, target_sr=16000)
            res = self.vad_model.generate(input=indata_16k, cache=self.vad_cache, is_final=False, chunk_size=self.vad_chunk_size)
            res_value = res[0]["value"]
            print(res_value)
            if len(res_value) % 2 == 1 and not self.vad_speech_detected:
                self.vad_speech_detected = True
            elif len(res_value) % 2 == 1 and self.vad_speech_detected:
                self.set_speech_detected_false_at_end_flag = True
            end_event.record()
            if device.type == "mps":
                torch.mps.synchronize()  # MPS - Wait for the events to be recorded!
            else:
                torch.cuda.synchronize()  # Wait for the events to be recorded!
            elapsed_time_ms = start_event.elapsed_time(end_event)
            print(f"Time taken for VAD: {elapsed_time_ms}ms")

            # if self.gui_config.threhold > -60:
            #     indata = np.append(self.rms_buffer, indata)
            #     rms = librosa.feature.rms(
            #         y=indata, frame_length=4 * self.zc, hop_length=self.zc
            #     )[:, 2:]
            #     self.rms_buffer[:] = indata[-4 * self.zc :]
            #     indata = indata[2 * self.zc - self.zc // 2 :]
            #     db_threhold = (
            #         librosa.amplitude_to_db(rms, ref=1.0)[0] < self.gui_config.threhold
            #     )
            #     for i in range(db_threhold.shape[0]):
            #         if db_threhold[i]:
            #             indata[i * self.zc : (i + 1) * self.zc] = 0
            #     indata = indata[self.zc // 2 :]
            self.input_wav[: -self.block_frame] = self.input_wav[
                self.block_frame :
            ].clone()
            self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(
                self.config.device
            )
            self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
                self.block_frame_16k :
            ].clone()
            self.input_wav_res[-320 * (indata.shape[0] // self.zc + 1) :] = (
                # self.resampler(self.input_wav[-indata.shape[0] - 2 * self.zc :])[
                #     320:
                # ]
                torch.from_numpy(librosa.resample(self.input_wav[-indata.shape[0] - 2 * self.zc :].cpu().numpy(), orig_sr=self.gui_config.samplerate, target_sr=16000)[320:])
            )
            print(f"preprocess time: {time.perf_counter() - start_time:.2f}")
            # infer
            if self.function == "vc":
                if self.gui_config.extra_time_ce - self.gui_config.extra_time < 0:
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
                    self.model_set,
                    self.reference_wav,
                    self.gui_config.reference_audio_path,
                    self.input_wav_res,
                    self.block_frame_16k,
                    self.skip_head,
                    self.skip_tail,
                    self.return_length,
                    int(self.gui_config.diffusion_steps),
                    self.gui_config.inference_cfg_rate,
                    self.gui_config.max_prompt_length,
                    self.gui_config.extra_time_ce - self.gui_config.extra_time,
                )
                if self.resampler2 is not None:
                    infer_wav = self.resampler2(infer_wav)
                end_event.record()
                if device.type == "mps":
                    torch.mps.synchronize()  # MPS - Wait for the events to be recorded!
                else:
                    torch.cuda.synchronize()  # Wait for the events to be recorded!
                elapsed_time_ms = start_event.elapsed_time(end_event)
                print(f"Time taken for VC: {elapsed_time_ms}ms")
                if not self.vad_speech_detected:
                    infer_wav = torch.zeros_like(self.input_wav[self.extra_frame :])
            elif self.gui_config.I_noise_reduce:
                infer_wav = self.input_wav_denoise[self.extra_frame :].clone()
            else:
                infer_wav = self.input_wav[self.extra_frame :].clone()

            # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
            conv_input = infer_wav[
                None, None, : self.sola_buffer_frame + self.sola_search_frame
            ]

            cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
            cor_den = torch.sqrt(
                F.conv1d(
                    conv_input**2,
                    torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device),
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
            infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
            infer_wav[: self.sola_buffer_frame] += (
                self.sola_buffer * self.fade_out_window
            )
            self.sola_buffer[:] = infer_wav[
                self.block_frame : self.block_frame + self.sola_buffer_frame
            ]
            outdata[:] = (
                infer_wav[: self.block_frame]
                .repeat(self.gui_config.channels, 1)
                .t()
                .cpu()
                .numpy()
            )

            total_time = time.perf_counter() - start_time
            if flag_vc:
                self.infer_time_label.setText(f'Inference Time: {int(total_time * 1000)}')
            if self.set_speech_detected_false_at_end_flag:
                self.vad_speech_detected = False
                self.set_speech_detected_false_at_end_flag = False
            print(f"Infer time: {total_time:.2f}")
        def sr(self, i):
            if not self.sr_flag: return
            if self.sr_model_checkbox.isChecked() and self.sr_device_checkbox.isChecked():
                if i==1: self.sr_device_checkbox.setChecked(False)
                else:  self.sr_model_checkbox.setChecked(False)
            elif not (self.sr_model_checkbox.isChecked() or self.sr_device_checkbox.isChecked()):
                if i==1: self.sr_device_checkbox.setChecked(True)
                else:  self.sr_model_checkbox.setChecked(True)
        def open_audio_file(self):
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select an Audio File",
                "",
                "Audio Files (*.mp3 *.wav *.ogg *.flac)"
            )
            if file_path:
                self.ref_audio_button.setText(f"{file_path}")   
        def hostapi_changed(self):
            hostapi = self.hostapi_dropdown.currentText()
            inputDevice = self.inputDevice_dropdown.currentText()
            outputDevice = self.outputDevice_dropdown.currentText()
            # Call logic depending on selected hostapi
            self.update_devices(hostapi_name=hostapi)

            # Clear and repopulate hostapi dropdown
            self.hostapi_dropdown.blockSignals(True)  # Prevent recursive signal triggering
            self.hostapi_dropdown.clear()
            self.hostapi_dropdown.addItems(self.hostapis)

            # Set default selection back to the same hostapi (if it exists)
            if hostapi in self.hostapis:
                self.hostapi_dropdown.setCurrentText(hostapi)
            else:
                self.hostapi_dropdown.setCurrentIndex(0)  # fallback to first item
            self.hostapi_dropdown.blockSignals(False)
            
            # Clear and repopulate inputDevice dropdown
            self.inputDevice_dropdown.blockSignals(True)  # Prevent recursive signal triggering
            self.inputDevice_dropdown.clear()
            self.inputDevice_dropdown.addItems(self.input_devices)

            if inputDevice in self.input_devices:
                self.inputDevice_dropdown.setCurrentText(inputDevice)
            else:
                self.inputDevice_dropdown.setCurrentIndex(0)  # fallback to first item
            self.inputDevice_dropdown.blockSignals(False)


            # Clear and repopulate outputDevice dropdown
            self.outputDevice_dropdown.blockSignals(True)  # Prevent recursive signal triggering
            self.outputDevice_dropdown.clear()
            self.outputDevice_dropdown.addItems(self.output_devices)

            if outputDevice in self.output_devices:
                self.outputDevice_dropdown.setCurrentText(outputDevice)
            else:
                self.outputDevice_dropdown.setCurrentIndex(0)  # fallback to first item
            self.outputDevice_dropdown.blockSignals(False)
        def update_devices(self, hostapi_name=None):
            """Get input and output devices."""
            global flag_vc
            flag_vc = False
            sd._terminate()
            sd._initialize()
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            for hostapi in hostapis:
                for device_idx in hostapi["devices"]:
                    devices[device_idx]["hostapi_name"] = hostapi["name"]
            self.hostapis = [hostapi["name"] for hostapi in hostapis]
            if hostapi_name not in self.hostapis:
                hostapi_name = self.hostapis[0]
            self.input_devices = [
                d["name"]
                for d in devices
                if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.output_devices = [
                d["name"]
                for d in devices
                if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.input_devices_indices = [
                d["index"] if "index" in d else d["name"]
                for d in devices
                if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]
            self.output_devices_indices = [
                d["index"] if "index" in d else d["name"]
                for d in devices
                if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
            ]





    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--config-path", type=str, default=None, help="Path to the vocoder checkpoint")
    parser.add_argument("--fp16", type=str2bool, nargs="?", const=True, help="Whether to use fp16", default=True)
    parser.add_argument("--gpu", type=int, help="Which GPU id to use", default=0)
    args = parser.parse_args()
    cuda_target = f"cuda:{args.gpu}" if args.gpu else "cuda" 

    if torch.cuda.is_available():
        device = torch.device(cuda_target)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = MainWindow(args)
    window.show()
    
    sys.exit(app.exec())


    