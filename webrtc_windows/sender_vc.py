import asyncio
import json
import os
import sounddevice as sd
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from collections import deque
import argparse

CHUNK_DURATION = 0.3  # seconds
SAMPLE_RATE = 48000
CHANNELS = 1
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)

parser = argparse.ArgumentParser(description='Live Voice Cloning')
parser.add_argument('--vb_audio', action='store_true', help='Send frames to virtual audio output')
args = parser.parse_args()

DEVICE = None
if args.vb_audio:
    for dev in sd.query_devices():
        if "VB-Cable" in dev["name"]:
            DEVICE = dev['index']
    if DEVICE is None:
        print("Virtual Output Audio not available. Please install VB Audio: https://vb-audio.com/Cable/")
        DEVICE= default_output = sd.default.device[1]  
else:
    DEVICE= default_output = sd.default.device[1]  

print(f"Device Index: {DEVICE}")

incoming_audio_chunks = deque() # buffer for processed audio



prev_talk=False
switch=0

async def capture_and_send(channel):
    audio_queue = asyncio.Queue(maxsize=14)
    def callback(indata, frames, time_info, status):
        global prev_talk, switch
        if status:
            print(f"Sounddevice input status: {status}")

        mono_audio = np.mean(indata, axis=1) if indata.ndim > 1 else indata
        rms = np.sqrt(np.mean(mono_audio**2))

        VAD_THRESHOLD = 0.005
        
        if rms < VAD_THRESHOLD and not prev_talk:
            return  
        
        if rms < VAD_THRESHOLD and prev_talk:
            switch+=1
        try:
            audio_queue.put_nowait(indata.copy())
            prev_talk=True
            if switch==3:
                switch=0
                prev_talk=False

        except asyncio.QueueFull:
            pass
    # Start stream
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32',
                        blocksize=CHUNK_SAMPLES, callback=callback):
        while True:
            chunk = await audio_queue.get()

            int16_audio = (chunk * 32767).astype(np.int16)
            channel.send(int16_audio.tobytes())

async def playback_processed_audio():
    def callback(outdata, frames, time_info, status):
        if status:
            print(f"Sounddevice output status: {status}")
        if incoming_audio_chunks:
            chunk = incoming_audio_chunks.popleft()
            data = np.frombuffer(chunk, dtype=np.int16).reshape(-1, CHANNELS)
            outdata[:len(data)] = data
        else:
            outdata.fill(0.0)

    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16',
                         blocksize=CHUNK_SAMPLES, callback=callback, device=DEVICE):

        await asyncio.Future()

async def run_peer_a():
    for file in ["offer.json", "answer.json"]:
        if os.path.exists(file):
            os.remove(file)

    config = RTCConfiguration([
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
    ])
    pc = RTCPeerConnection(configuration=config)

    channel = pc.createDataChannel("audio")

    @channel.on("open")
    def on_open():
        print("DataChannel is open")

    @channel.on("message")
    def on_message(message):
        if isinstance(message, bytes):
            print(f"Received processed audio chunk of size {len(message)}")
            incoming_audio_chunks.append(message)
        else:
            print(f"⚠️ Unexpected message: {message}")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "disconnected"):
            await pc.close()

    print("Creating offer...")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    print("Waiting for ICE gathering to complete...")
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.1)

    offer_data = {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }

    with open("offer.json", "w") as f:
        json.dump(offer_data, f, indent=2)

    print("\nSTEP 1 complete: Offer and ICE candidates created.")
    print("Run Peer B, wait for answer.json, then press Enter to continue.")
    input("Press Enter when ready...")

    while not os.path.exists("answer.json"):
        await asyncio.sleep(1)
        print("Waiting for answer.json...")

    with open("answer.json", "r") as f:
        answer_data = json.load(f)

    answer = RTCSessionDescription(
        sdp=answer_data["sdp"],
        type=answer_data["type"]
    )
    await pc.setRemoteDescription(answer)
    print("📥 Remote description (answer) set")

    try:
        await asyncio.gather(
            capture_and_send(channel),
            playback_processed_audio()
        )
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        await pc.close()

if __name__ == "__main__":
    asyncio.run(run_peer_a())