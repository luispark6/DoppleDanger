# How to Use Webrtc for Remote Live Swapping for Macbook
### Use Case:
This is very similar to the windows implementation. However, this does NOT send back frames to the macbook. Rather, it streams the audio and frames to the Windows desktop's virtual camera and audio. Then, you can remotely use this deepfake by remotely accessing the Window's chrome browser, and joining a virtual meeting. This can be easily done using the remote chrome functionaility. This takes about 2-3 steps, please search up online how to do so.

### Instructions:
# System Setup Overview

* **Machine A** – This is the powerful machine where live **face swapping** or **voice cloning** will occur. It must have a capable **GPU** to handle intensive post-processing.
* **Machine B** – This machine captures the live webcam feed or audio, and **sends video/audio frames to Machine A**. It requires minimal GPU resources.

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
pip install aiortc opencv-python sounddevice
```

* Run the sender script for either the voice conversion or face swapping (***from Machine B obviously***):

```bash
python sender_fs.py   ##face swapping functionality
python sender_fs.py --obs  ##outputting the frames to a virtual camera on machine B 
```
OR  
```bash
python sender_vc.py   ##voice cloning functionality
python sender_vc.py --vb_audio  ##outputting the audio to a virtual audio on machine B
``` 

NOTE YOU MUST INSTALL **OBS** AND **VB_AUDIO** FOR VIRTUAL OUPUTS

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
python receiver_fs.py --source /path/to/ref_img.png --modelPath /path/to/model.pth ##This is for face swapping
```
OR
```bash
python receiver_vc.py --reference_path /path/to/ref_img.png ##This is for voice cloning
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

## 6. Chrome Remote Browser
* For macbook, since you can not stream faceswapping to virtual cam, this implementation will be sending the frames to **Machine B's** virtual camera and audio 
* So what you must do is install chrome, and remotely access **Machine B's** chrome browser and join virtual meeting there. This is incredibly easy using
the remote chrome application


## 6. Limitations
* `sender_vc.py` and `receiver_vc.py` have hard-coded sample rates, frame sizes, etc.. Not dynamic to any audio input and ouput device
* Voice conversion delay is quite high due to large audio chunk size requirements