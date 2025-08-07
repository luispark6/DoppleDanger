import asyncio
import cv2
import json
import os
import time
from aiortc import RTCPeerConnection, VideoStreamTrack, RTCConfiguration, RTCIceServer, RTCSessionDescription
from av import VideoFrame
import argparse
import pyvirtualcam
from pyvirtualcam import PixelFormat
import fractions
from collections import deque
from contextlib import nullcontext
import numpy as np

parser = argparse.ArgumentParser(description='Live face swap via webcam')

args = parser.parse_args()




class CameraStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

        self.buffer = deque()
        cv2.namedWindow('Delay Control')
        cv2.createTrackbar('Delay (ms)', 'Delay Control', 1, 10000, lambda x: None)
        cv2.resizeWindow('Delay Control', 500, 2)  # Optional initial size


        self.target_fps = 13
        self.last_frame_time = time.time()
        self.delay=1

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
    



        if cv2.getWindowProperty('Delay Control', cv2.WND_PROP_VISIBLE) < 1:
            cv2.namedWindow('Delay Control')
            cv2.createTrackbar('Delay (ms)', 'Delay Control', self.delay, 10000, lambda x: None)
            cv2.resizeWindow('Delay Control', 500, 2)


        frame = cv2.resize(frame, (960, 540))
        cv2.waitKey(1)  # <-- Add this line




        self.delay = cv2.getTrackbarPos('Delay (ms)', 'Delay Control')
        buffer_end = time.time()
        self.buffer.append((frame, buffer_end))

        if (buffer_end-self.buffer[0][1])*1000>= self.delay:

            new_frame = VideoFrame.from_ndarray(self.buffer[0][0], format="bgr24")
            self.buffer.popleft()
        else:
            # Not enough time — show a black screen
            black_frame = np.zeros((540, 960, 3), dtype=np.uint8)
            new_frame = VideoFrame.from_ndarray(black_frame, format="bgr24")

        

        new_frame.pts = pts     

        elapsed = time.time() - self.last_frame_time
        target_duration  = 1/self.target_fps

        if elapsed<target_duration:
            await asyncio.sleep(target_duration-elapsed)

        self.last_frame_time = time.time()

        new_frame.time_base  = fractions.Fraction(1, self.target_fps)


        return new_frame

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()


async def run_peer_a():
    for file in ["offer.json", "answer.json"]:
        if os.path.exists(file):
            os.remove(file)

    config = RTCConfiguration([
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
    ])
    pc = RTCPeerConnection(configuration=config)
    local_video = CameraStreamTrack()
    pc.addTrack(local_video)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state: {pc.connectionState}")

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print(f"ICE connection state: {pc.iceConnectionState}")

    @pc.on("icegatheringstatechange")
    async def on_icegatheringstatechange():
        print(f"ICE gathering state: {pc.iceGatheringState}")



    pc.addTransceiver("video", direction="sendrecv")

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

    print("\nSTEP 1: Offer and ICE candidates created!")
    print("Now run receiver.py, wait for it to generate answer.json, then press Enter.")
    input()

    while not os.path.exists("answer.json"):
        await asyncio.sleep(1)
        print("Still waiting for answer.json...")

    with open("answer.json", "r") as f:
        answer_data = json.load(f)

    answer = RTCSessionDescription(
        sdp=answer_data["sdp"],
        type=answer_data["type"]
    )
    await pc.setRemoteDescription(answer)
    print("Remote description (answer) set")

    try:
        while True:
            await asyncio.sleep(5)
            print(f"Status - Connection: {pc.connectionState}, ICE: {pc.iceConnectionState}")
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        await pc.close()
        local_video.cap.release()

if __name__ == "__main__":
    asyncio.run(run_peer_a())