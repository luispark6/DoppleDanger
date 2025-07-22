# How to Use Webrtc for Remote Live Swapping
### Use Case:
Offloading the GPU workload of the faceswap to a remote machine. For example, suppose my powerful desktop is at home, and I am in a different location with my laptop. I want to run the live face swap on my laptop, but my laptop's gpu is far too weak to run it. What we can do is offload the faceswap workload to the desktop at home, then send the faceswapped frames back to my laptop. To do this, we need to use a peer2peer connection for the fastest and most efficient source of communication. We use webrtc to accomplish this.

### Instructions:
# System Setup Overview

* **Machine A** – This is the powerful machine where live **face swapping** will occur. It must have a capable **GPU** to handle intensive post-processing.
* **Machine B** – This machine captures the live webcam feed, **sends video frames to Machine A**, and **receives the processed frames** back. It requires minimal GPU resources.

# Setup Instructions

## 0. Ensure SSH Access

First, make sure **Machine B can SSH into Machine A**. This may involve:

* Installing **OpenSSH Server** on Machine A
* Configuring the **firewall to allow port 22**
* Setting up **port forwarding** on your router if connecting over different networks

There are many detailed tutorials online for setting up SSH on Windows/Linux.

## 1. Prepare Machine B (Sender Side)

* Copy the `sender.py` file to **Machine B**
* Install required dependencies:

```bash
pip install aiortc opencv-python
```

* Run the sender script (***from Machine B obviously***):

```bash
python sender.py
```

## 2. Transfer `offer.json` to Machine A

* After running `sender.py`, a file named `offer.json` will be created
* Transfer it to the correct directory on Machine A:

```bash
scp ./path/to/offer.json yourname@<your-home-public-ip>:/path/to/DoppleDanger/webrtc/
```

## 3. Run the Receiver on Machine A

* SSH into Machine A from Machine B:

```bash
ssh yourname@<your-home-public-ip>
```

* Navigate to the `DoppleDanger/webrtc` directory:

```bash
cd /path/to/DoppleDanger/webrtc
```

* Start the receiver:

```bash
python receiver.py
```

## 4. Transfer `answer.json` Back to Machine B

* After running `receiver.py`, a file named `answer.json` will appear in the same directory
* Transfer this file from Machine A back to Machine B:

```bash
scp yourname@<your-home-public-ip>:/path/to/DoppleDanger/webrtc/answer.json ./path/to/sender.py/folder/
```

## 5. Complete Handshake

* Go back to the terminal on **Machine B** where `sender.py` is running
* Press **Enter** to complete the WebRTC handshake and begin streaming