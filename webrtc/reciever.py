# peer_b_manual.py - The receiver/processor (run this after peer_a creates offer.json)
import asyncio
import cv2
import json
import os
import time
from aiortc import RTCPeerConnection, VideoStreamTrack, RTCConfiguration, RTCIceServer, RTCSessionDescription, RTCIceCandidate
from av import VideoFrame

class ProcessingStreamTrack(VideoStreamTrack):
    def __init__(self, original_track):
        super().__init__()
        self.track = original_track

    async def recv(self):
        frame = await self.track.recv()
        
        # Convert to numpy array for processing
        # img = frame.to_ndarray(format="bgr24")
        
        # # Apply grayscale processing
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # # Convert back to VideoFrame
        # new_frame = VideoFrame.from_ndarray(gray_bgr, format="bgr24")
        # new_frame.pts = frame.pts
        # new_frame.time_base = frame.time_base

        ####THIS IS WHERE FACE SWAP OCCURS

        return frame

async def run_peer_b():
    # Wait for offer file
    print("Waiting for offer.json from Peer A...")
    while not os.path.exists("offer.json"):
        await asyncio.sleep(1)
        print("Still waiting for offer.json...")
    
    print("Found offer.json!")
    
    # WebRTC configuration  
    config = RTCConfiguration([
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
    ])
    pc = RTCPeerConnection(configuration=config)
    
   
    
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
        print(f"Received track: {track.kind}")
        if track.kind == "video":
            # Create processed track and add it back
            processed_track = ProcessingStreamTrack(track)
            pc.addTrack(processed_track)
            print("Added processed video track")


    # Load and set offer
    with open("offer.json", "r") as f:
        offer_data = json.load(f)
    
    offer = RTCSessionDescription(
        sdp=offer_data["sdp"],
        type=offer_data["type"]
    )
    await pc.setRemoteDescription(offer)
    print("Remote description (offer) set")
    
    
   
    
    # Create answer
    print("Creating answer...")
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    # Wait for ICE gathering to complete
    print("Waiting for ICE gathering to complete...")
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.1)
    
    # Save answer and ICE candidates
    answer_data = {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }
    
    with open("answer.json", "w") as f:
        json.dump(answer_data, f, indent=2)
    

    
    print(f"\n{'='*50}")
    print("STEP 2: Answer and ICE candidates created!")
    print(f"{'='*50}")
    print("Files created: answer.json")
    print("\nGo back to Peer A and press Enter to continue the connection")
    
    print("\nConnection setup complete! Monitoring connection...")
    print("Press Ctrl+C to stop")
    
    # Keep running and monitor connection
    try:
        while True:
            await asyncio.sleep(5)
            print(f"Status - Connection: {pc.connectionState}, ICE: {pc.iceConnectionState}")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await pc.close()

if __name__ == "__main__":
    asyncio.run(run_peer_b())