# Overview Documentation
Explaining broad overview of the live voice cloning and faceswapping mechanisms.

## `swap_live_video.py`
This is where the main faceswapping mechanism takes place. Here, is the pipeline:

1. **Create OpenCV Window**  
   - Initialize a `cv2` window to display results.

2. **Capture Target Face from Webcam**  
   - Grab a frame from the webcam containing the target face.

3. **Color Transfer**  (Only Once)
   - Apply color transfer from the target face to the source image for better blending.

4. **Extract Latent Vectors (Source Image)**  
   - Use the **`buffalo_l`** model to extract latent vectors from the source image.

5. **Preprocess Target Image**  
   - Apply preprocessing steps such as:
     - `warpAffine`  
     - `getBlob`  
     - Other normalization or alignment steps

6. **Face Swap (Reswapper Model)**  
   - Pass the preprocessed target image and the latent vectors into the **Reswapper** model to perform the face swap.

7. **Postprocess Output**  
   - Improve cohesion of the swapped face (smoothing, blending, etc.).
   - Send the final frame to:
     - The OpenCV display window, or  
     - OBS for streaming.

8. **Repeat**  
   - Return to **Step 2** to continuously capture and swap in real-time.


## `unified_gui.py`
This GUI provides an easy way to **run face swapping and voice cloning simultaneously** — no need to juggle multiple terminal commands.  

- **Face Swapping**  
  Uses `swap_live_video.py` to perform real-time face swapping from your webcam feed.  

- **Voice Cloning**  
  - Initializes the necessary parameters for voice conversion.  
  - Captures microphone input via `sounddevice`.  
  - Streams audio into an `audiocallback` function, which:
    1. Preprocesses the incoming audio.
    2. Sends it to the model for real-time voice conversion.  

For a deep dive into the voice cloning pipeline, check out the excellent [seed-vc repository](https://github.com/Plachtaa/seed-vc).  

**In short:** This GUI is focused on usability — giving you a streamlined, click-to-run interface instead of a purely command-line workflow.


## `train.py`
Uses the `StyleTransferModel_128.py` architecture to train the **Reswapper** model.  
This is a **supervised learning** setup where the model’s outputs are compared against those of the well-known **InSwapper** model to calculate the training loss.  

In addition to this style transfer loss, an **identity loss** is incorporated to help preserve the original subject’s facial features during swapping.

## `StyleTransferModel_128.py`

At a high level, the swapping model takes the **latent vector** of the source image and the **target image** as inputs.  
It then injects **style blocks** derived from the source latent vector into the target image.  

The style blocks carry the **identity features** of the source face, while the model aims to preserve the **lighting, color tone, and other contextual details** of the target image for a natural-looking result. 

A major drawback of this model is that it tends to **over-smooth skin**, causing the swapped face to lose much of the original skin texture and some of the target face’s lighting.  
This can give the output a slightly **animated or artificial look**.

To address this, I have experimented with several approaches, including:  
- Adding a **discriminator** during training to improve facial realism  (used the discriminator found in `iresnet.py`)
- Introducing **skip connections** to preserve fine details  
- Integrating **GFPGAN** for texture enhancement  
- Incorporating additional **loss functions** to encourage skin texture retention  

Unfortunately, none of these methods have produced a satisfactory fix so far.

## `Image.py` and `face_align.py`
Both files primarily used in `swap_live_video.py` for preprocessing and postprocessing the target and source images. 

## `faceswap_gui.py` and `voice_clone_gui.py`
Seperate GUIs for faceswapping and voicecloning. If for some reason you don't want a GUI with both functionalities, these can be used. These GUIs have very similar formatting to the unified GUI. 