# peer_a_manual.py - The sender (run this first)
import asyncio
import cv2
import json
import time
import os
from aiortc import RTCPeerConnection, VideoStreamTrack, RTCConfiguration, RTCIceServer, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.signaling import object_to_string, object_from_string
from aiortc.contrib.media import MediaRecorder
from av import VideoFrame
i=0
class CameraStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

    async def recv(self):
        
        pts, time_base = await self.next_timestamp()
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        
        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        return new_frame

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

async def run_peer_a():
    # Clean up old files
    for file in ["offer.json", "answer.json"]:
        if os.path.exists(file):
            os.remove(file)
    
  

    # WebRTC configuration
    config = RTCConfiguration([
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
    ])
    pc = RTCPeerConnection(configuration=config)
    
    # Add local video track
    local_video = CameraStreamTrack()
    pc.addTrack(local_video)
    
    # Store ICE candidates
    
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
            # Start a new async task to read and display frames
            async def display_frames():
                fps=0
                fps_update_interval = 0.5
                frame_count = 0
                prev_time = time.time()
                while True:
                    try:
                        frame = await track.recv()
                        current_time = time.time()
                        frame_count += 1
                        if current_time - prev_time >= fps_update_interval:
                            fps = frame_count / (current_time - prev_time)
                            frame_count = 0
                            prev_time = current_time
                        # print(fps)
                        img = frame.to_ndarray(format="bgr24")
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
                        cv2.imshow("Received Video", img)
                        
                        # Press 'q' to quit display window
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except Exception as e:
                        print(f"Error displaying frame: {e}")
                        break

                cv2.destroyAllWindows()

            # Launch the async display loop
            asyncio.ensure_future(display_frames())


    pc.addTransceiver("video", direction="sendrecv")
    # pc.addTransceiver("video", direction="recvonly")  # For receiving processed video


    # Create and save offer
    print("Creating offer...")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    
    # Wait for ICE gathering to complete
    print("Waiting for ICE gathering to complete...")
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.1)



    # Save offer and ICE candidates
    offer_data = {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }
    
    with open("offer.json", "w") as f:
        json.dump(offer_data, f, indent=2)

    
    print(f"\n{'='*50}")
    print("STEP 1: Offer and ICE candidates created!")
    print(f"{'='*50}")
    print("File created: offer.json")
    print("\nNow run peer_b_manual.py")
    print("Wait for it to create answer.json")
    print("Then press Enter to continue...")
    input()
    
    # Wait for answer file
    print("Waiting for answer file...")
    while not os.path.exists("answer.json"):
        await asyncio.sleep(1)
        print("Still waiting for answer.json...")
    
    # Load and set answer
    with open("answer.json", "r") as f:
        answer_data = json.load(f)
    
    answer = RTCSessionDescription(
        sdp=answer_data["sdp"],
        type=answer_data["type"]
    )
    await pc.setRemoteDescription(answer)
    print("Remote description (answer) set")
    
   
    
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
        local_video.cap.release()

if __name__ == "__main__":
    asyncio.run(run_peer_a())

