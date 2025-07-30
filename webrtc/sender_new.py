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

from contextlib import nullcontext

parser = argparse.ArgumentParser(description='Live face swap via webcam')
parser.add_argument('--obs',action='store_true', help='Send frames to obs virtual cam')

args = parser.parse_args()




class CameraStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

        self.target_fps = 13
        self.last_frame_time = time.time()

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")

        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
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

    @pc.on("track")
    def on_track(track):
        print(f"Receiving processed track: {track.kind}")
        if track.kind == "video":
            async def display_frames():
                fps = 0
                frame_count = 0
                prev_time = time.time()
                try:
                    with pyvirtualcam.Camera(width=640, height=480, fps=13, fmt=PixelFormat.BGR) if args.obs else nullcontext() as cam:
                        while True:
                            frame = await track.recv()
                            img = frame.to_ndarray(format="bgr24")

                            current_time = time.time()
                            frame_count += 1
                            if current_time - prev_time >= 0.5:
                                fps = frame_count / (current_time - prev_time)
                                frame_count = 0
                                prev_time = current_time

                            cv2.putText(
                                img,
                                f"FPS: {fps:.2f}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA
                            )
                            if cam:
                                cam.send(img)
                                cam.sleep_until_next_frame()
                            else:
                                cv2.imshow("Received Video", img)


                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                except Exception as e:
                    print(f"[Receiver error] {e}")
                finally:
                    cv2.destroyAllWindows()

            asyncio.ensure_future(display_frames())

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