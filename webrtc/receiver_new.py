import asyncio
import cv2
import json
import os
import time
import numpy as np
from aiortc import RTCPeerConnection, VideoStreamTrack, RTCConfiguration, RTCIceServer, RTCSessionDescription
from av import VideoFrame


class ProcessingStreamTrack(VideoStreamTrack):
    def __init__(self, original_track):
        super().__init__()
        self.track = original_track

    async def recv(self):
        try:
            frame = await self.track.recv()
            # Placeholder for future face-swap or image manipulation
            return frame
        except Exception as e:
            print(f"[Processing recv error] {e}")
            await asyncio.sleep(1 / 30)
            return VideoFrame.from_ndarray(np.zeros((480, 640, 3), dtype=np.uint8), format="bgr24")


async def run_peer_b():
    print("Waiting for offer.json from Peer A...")
    while not os.path.exists("offer.json"):
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
            # ProcessingStreamTrack initialized but NOT sent back to sender
            processed_track = ProcessingStreamTrack(track)
            async def display_processed_frames():
                try:
                    while True:
                        frame = await processed_track.recv()
                        img = frame.to_ndarray(format="bgr24")
                        cv2.imshow("Processed Video", img)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                except Exception as e:
                    print(f"[Display error] {e}")
                finally:
                    cv2.destroyAllWindows()

            asyncio.ensure_future(display_processed_frames())

    with open("offer.json", "r") as f:
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

    with open("answer.json", "w") as f:
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
