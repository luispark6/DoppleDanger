import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

print(parent_dir)
os.chdir(parent_dir)

import asyncio
import cv2
import json
import time
import numpy as np
from aiortc import RTCPeerConnection, VideoStreamTrack, RTCConfiguration, RTCIceServer, RTCSessionDescription
from av import VideoFrame
import argparse
import cv2
import numpy as np
import torch
import Image
from insightface.app import FaceAnalysis
import face_align
from StyleTransferModel_128 import StyleTransferModel
# from StyleTransferModel_skip import StyleTransferModel_skip
import time 




faceAnalysis = FaceAnalysis(name='buffalo_l')
faceAnalysis.prepare(ctx_id=0, det_size=(512, 512))

parser = argparse.ArgumentParser(description="Live face swap WebRTC")
parser.add_argument('--source', required=True, help='Path to source face image')
parser.add_argument('--modelPath', required=True, help='Path to the trained face swap model')
args = parser.parse_args()

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    device = get_device()
    model = StyleTransferModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False), strict=False)
    model.eval()
    return model


def create_source_latent(source_image, direction_path=None, steps=0.0):
    faces = faceAnalysis.get(source_image)
    if len(faces) == 0:
        print("No face detected in source image.")
        return None

    source_latent = Image.getLatent(faces[0])
    if direction_path:
        direction = np.load(direction_path)
        direction = direction / np.linalg.norm(direction)
        source_latent += direction * steps
    return source_latent

def apply_color_transfer(source_path, target):
    """
    Apply color transfer from target to source image
    """

    source = cv2.imread(source_path)
    target_faces = faceAnalysis.get(target)
    x1, y1, x2, y2= target_faces[0]["bbox"]
    target = target[int(y1): int(y2), int(x1):int(x2)]
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    source_mean, source_std = cv2.meanStdDev(source)
    target_mean, target_std = cv2.meanStdDev(target)

    # Reshape mean and std to be broadcastable
    source_mean = source_mean.reshape(1, 1, 3)
    source_std = source_std.reshape(1, 1, 3)
    target_mean = target_mean.reshape(1, 1, 3)
    target_std = target_std.reshape(1, 1, 3)

    # Perform the color transfer
    source = (source - source_mean) * (target_std / source_std) + target_mean

    return cv2.cvtColor(np.clip(source, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)



def swap_face(model, target_face, source_face_latent):
    device = get_device()
    target_tensor = torch.from_numpy(target_face).to(device)
    source_tensor = torch.from_numpy(source_face_latent).to(device)

    with torch.no_grad():
        swapped_tensor = model(target_tensor, source_tensor)

    swapped_face = Image.postprocess_face(swapped_tensor)
    return swapped_face


import fractions
model = load_model(args.modelPath)

class ProcessingStreamTrack(VideoStreamTrack):
    def __init__(self, original_track):
        super().__init__()
        self.track = original_track
        self.source_latent = None
        self.create_latent_flag = True
        self.counter = 0
        self.last_frame_time = time.time()

        

    async def recv(self):
        try:
            vf = await self.track.recv()

            frame = vf.to_ndarray(format="bgr24")
    
            if self.create_latent_flag:
                source = apply_color_transfer(source_path=args.source, target= frame)
                self.source_latent = create_source_latent(source, None, 0.0)
                if self.source_latent is None:
                    await asyncio.sleep(1 / 13)
                    print("Source Face could not be found")
                    return frame
                
                self.create_latent_flag = False


            faces = faceAnalysis.get(frame)
            if len(faces) == 0:
                return frame
            target_face = faces[0]

             # x1,y1,x2,y2 = target_face['bbox']
            aligned_face, M = face_align.norm_crop2(frame, target_face.kps, 128)
            face_blob = Image.getBlob(aligned_face, (128, 128))


            swapped_face = swap_face(model, face_blob, self.source_latent)
            final_frame = Image.blend_swapped_image_gpu(swapped_face, frame, M)
            # Placeholder for future face-swap or image manipulation

            video_frame = VideoFrame.from_ndarray(final_frame, format="bgr24")
            video_frame.pts = self.counter
            video_frame.time_base = fractions.Fraction(1, 13)
            self.counter += 1
            return video_frame
        
        except Exception as e:
            print(f"[Processing recv error] {e}")
            await asyncio.sleep(1 / 13)
            return VideoFrame.from_ndarray(np.zeros((480, 640, 3), dtype=np.uint8), format="bgr24")


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
            print("Connection lost. Consider restarting peer.")

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print(f"ICE connection state: {pc.iceConnectionState}")

    @pc.on("icegatheringstatechange")
    async def on_icegatheringstatechange():
        print(f"ICE gathering state: {pc.iceGatheringState}")

    @pc.on("track")
    def on_track(track):
        print(f"Received track: {track.kind}")
        if track.kind == "video":
            # Create processed track and add it back
            processed_track = ProcessingStreamTrack(track)
            pc.addTrack(processed_track)
            print("Added processed video track")

    with open("./webrtc_windows/offer.json", "r") as f:
        offer_data = json.load(f)

    offer = RTCSessionDescription(
        sdp=offer_data["sdp"],
        type=offer_data["type"]
    )
    await pc.setRemoteDescription(offer)
    print("Remote description (offer) set")

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

    print("\nSTEP 2: Answer and ICE candidates created!")
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
