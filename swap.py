import argparse
import os
import cv2
import numpy as np
import torch
import Image
from insightface.app import FaceAnalysis
import face_align
from StyleTransferModel_128 import StyleTransferModel
import time 
import line_profiler
import pyvirtualcam
from pyvirtualcam import PixelFormat
from contextlib import nullcontext

# Setup face detector
faceAnalysis = FaceAnalysis(name='buffalo_l')
faceAnalysis.prepare(ctx_id=0, det_size=(512, 512))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Live face swap via webcam')
    parser.add_argument('--source', required=True, help='Path to source face image')
    parser.add_argument('--modelPath', required=True, help='Path to the trained face swap model')
    parser.add_argument('--resolution', type=int, default=128, help='Resolution of the face crop')
    parser.add_argument('--face_attribute_direction', default=None, help='Path to face attribute direction.npy')
    parser.add_argument('--face_attribute_steps', type=float, default=0.0, help='Amount to move in attribute direction')
    parser.add_argument('--obs', type=bool, default=False, help='Send frames to obs virtual cam')
    parser.add_argument('--minimal', type=bool, default=False, help='If minimal face blending is wanted')



    return parser.parse_args()

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    device = get_device()
    model = StyleTransferModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    return model

def swap_face(model, target_face, source_face_latent):
    device = get_device()
    target_tensor = torch.from_numpy(target_face).to(device)
    source_tensor = torch.from_numpy(source_face_latent).to(device)

    with torch.no_grad():
        swapped_tensor = model(target_tensor, source_tensor)

    swapped_face = Image.postprocess_face(swapped_tensor)
    return swapped_face

def create_source_latent(source_img_path, direction_path=None, steps=0.0):
    source_image = cv2.imread(source_img_path)
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

# @line_profiler.profile
def main():
    args = parse_arguments()
    model = load_model(args.modelPath)
    source_latent = create_source_latent(args.source, args.face_attribute_direction, args.face_attribute_steps)

    if source_latent is None:
        return

    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    fps=0
    fps_update_interval = 0.5
    frame_count = 0
    cv2.namedWindow('Live Face Swap', cv2.WINDOW_NORMAL)  # <--- This makes window resizable


    print("Press 'q' to quit.")
    prev_time = time.time()

    print("Starting live face swap. Press 'q' to quit.")
    with pyvirtualcam.Camera(width=960, height=540, fps=20, fmt=PixelFormat.BGR) if args.obs else nullcontext() as cam:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= fps_update_interval:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time
            print(fps)

            faces = faceAnalysis.get(frame)
            if len(faces) == 0:
                cv2.imshow('Live Face Swap', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            target_face = faces[0]
            x1,y1,x2,y2 = target_face['bbox']
            aligned_face, M = face_align.norm_crop2(frame, target_face.kps, args.resolution)
            face_blob = Image.getBlob(aligned_face, (args.resolution, args.resolution))

            try:
                swapped_face = swap_face(model, face_blob, source_latent)
                final_frame = Image.blend_swapped_image_gpu(swapped_face, frame, M, args.minimal)
                # cv2.rectangle(final_frame,(int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)
                frame[int(y1):int(y2), int(x1):int(x2)] = final_frame[int(y1):int(y2), int(x1):int(x2)]
                if cam:
                    output_img = cv2.resize(frame, (960, 540))
                    cam.send(output_img)
                    cam.sleep_until_next_frame()
                else:
                    cv2.imshow('Live Face Swap', cv2.resize(frame, (960, 540)))
            except Exception as e:
                print(f"Swap error: {e}")
                cv2.imshow('Live Face Swap', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
