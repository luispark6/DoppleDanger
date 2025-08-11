# Navigation Guide

This section explains where to find and modify specific parts of the project.

## Image Capture & Face Swapping
- **`swap_live_video.py`** – Handles the entire pipeline for:
  - Capturing images
  - Preprocessing & postprocessing
  - Performing face swapping
  - Displaying the swapped image

## Image Processing Functions
- **`Image.py`** & **`face_align.py`** – Add or modify preprocessing and postprocessing functions here, then use them in `swap_live_video.py`.

## Face Swapping Model
- **`StyleTransferModel_128.py`** – Contains the architecture for the face swapping model.

## Model Training
- **`train.py`** – Defines the training process for the face swapping model.

## Unified GUI (Face Swapping & Voice Cloning)
- **`unified_gui.py`** – Controls both the GUI and voice cloning pipeline:
  - **GUI layout and controls**
  - **Voice cloning mechanism**
  - **`audiocallback`** – Preprocessing, model execution, and postprocessing for voice cloning
  - **`custom_infer`** – Inference logic for the voice cloning model
  - **`load_models`** – Initialization and parameter setup for the live voice cloning model
## Voice Cloning Functions
- I am not too familiar on how exactly the voice cloning mechanism works. However, I do know that much of the functionality of how and what models are being used is located in the `seedvc/modules` folder.

## Remote Usage
- **`webrtc_macbook/`** & **`webrtc_windows/`** – Code for running the application remotely on macOS or Windows.

