import argparse
import os
import cv2
import numpy as np
import torch
import Image
from insightface.app import FaceAnalysis
import face_align
from StyleTransferModel_128 import StyleTransferModel
# from StyleTransferModel_skip import StyleTransferModel_skip
import time 
import line_profiler
import pyvirtualcam
from pyvirtualcam import PixelFormat
from contextlib import nullcontext
from collections import deque
import traceback


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
    parser.add_argument('--obs', action='store_true', help='Send frames to obs virtual cam')
    parser.add_argument('--mouth_mask', action='store_true', help='Retain target mouth')
    parser.add_argument('--delay', type=int, default=0, help='delay time in milliseconds')
    parser.add_argument('--fps_delay', action='store_true', help='show fps and delay time on top corner')
    parser.add_argument('--enhance_res', action='store_true', help='Increase webcam resolution to 1920x1080')
    return parser.parse_args()

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    device = get_device()
    model = StyleTransferModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False), strict=False)
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

def create_source_latent(source_image, direction_path=None, steps=0.0):
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

def apply_color_transfer(source_path, target):
    """
    Apply color transfer from target to source image
    """

    source = cv2.imread(source_path)
    target_faces = faceAnalysis.get(target)
    x1, y1, x2, y2= target_faces[0]["bbox"]
    target = target[int(y1): int(y2), int(x1):int(x2)]
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    source_mean, source_std = cv2.meanStdDev(source)
    target_mean, target_std = cv2.meanStdDev(target)

    # Reshape mean and std to be broadcastable
    source_mean = source_mean.reshape(1, 1, 3)
    source_std = source_std.reshape(1, 1, 3)
    target_mean = target_mean.reshape(1, 1, 3)
    target_std = target_std.reshape(1, 1, 3)

    # Perform the color transfer
    source = (source - source_mean) * (target_std / source_std) + target_mean

    return cv2.cvtColor(np.clip(source, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)


def blend(source_latent, target_latent, alpha):
    return alpha*target_latent + (1-alpha)*source_latent

def main():
    args = parse_arguments()
    model = load_model(args.modelPath)
    cap = cv2.VideoCapture(0)  # Open webcam
    width = 640*2
    height = 480*2
    if args.enhance_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        width = 1920*2
        height = 1080*2
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    fps=0
    fps_update_interval = 0.5
    frame_count = 0
    cv2.namedWindow('Live Face Swap', cv2.WINDOW_NORMAL) 
    cv2.namedWindow('Delay Control')
    cv2.createTrackbar('Delay (ms)', 'Delay Control', args.delay, 10000, lambda x: None)
    cv2.resizeWindow('Delay Control', 500, 2)  # Optional initial size

    cv2.namedWindow('Target Retention')
    cv2.createTrackbar('TR %', 'Target Retention', 0, 100, lambda x: None)
    cv2.resizeWindow('Target Retention', 500, 2)  


    create_latent_flag = True
    prev_time = time.time()
    buffer= deque()

    alpha = 0
    

    with pyvirtualcam.Camera(width=width, height=height, fps=20, fmt=PixelFormat.BGR) if args.obs else nullcontext() as cam:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            


            # Reopen delay control window if closed
            if cv2.getWindowProperty('Delay Control', cv2.WND_PROP_VISIBLE) < 1:
                cv2.namedWindow('Delay Control')
                cv2.createTrackbar('Delay (ms)', 'Delay Control', args.delay, 10000, lambda x: None)
                cv2.resizeWindow('Delay Control', 500, 2)

            if cv2.getWindowProperty('Target Retention', cv2.WND_PROP_VISIBLE) < 1:
                cv2.namedWindow('Target Retention')
                cv2.createTrackbar('TR %', 'Target Retention', alpha, 100, lambda x: None)
                cv2.resizeWindow('Target Retention', 500, 2)  
                

            if cv2.getWindowProperty('Live Face Swap', cv2.WND_PROP_VISIBLE) < 1:
                cv2.namedWindow('Live Face Swap', cv2.WINDOW_NORMAL)

                        
            if create_latent_flag:
                try:
                    source = apply_color_transfer(source_path=args.source, target= frame)
                    orig_source_latent = create_source_latent(source, args.face_attribute_direction, args.face_attribute_steps)
                    if orig_source_latent is None:
                        return "Face not found in source image"
                    source_latent=orig_source_latent
                    create_latent_flag = False
                except:
                    cv2.imshow('Live Face Swap', frame)

                
            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= fps_update_interval:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time

            faces = faceAnalysis.get(frame)
            if len(faces) == 0:
                cv2.imshow('Live Face Swap', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            target_face = faces[0]

            # x1,y1,x2,y2 = target_face['bbox']
            aligned_face, M = face_align.norm_crop2(frame, target_face.kps, args.resolution)
            face_blob = Image.getBlob(aligned_face, (args.resolution, args.resolution))

            try:
                swapped_face = swap_face(model, face_blob, source_latent)
                final_frame = Image.blend_swapped_image_gpu(swapped_face, frame, M)
                if args.mouth_mask:
                    face_mask = Image.create_face_mask(target_face, frame)
                    _, mouth_cutout, mouth_box, lower_lip_polygon = (
                        Image.create_lower_mouth_mask(target_face, frame)
                    )
                    final_frame = Image.apply_mouth_area(
                        final_frame, mouth_cutout, mouth_box, face_mask, lower_lip_polygon
                    )

                # cv2.rectangle(final_frame,(int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)
              
                if args.fps_delay:
                    # Overlay FPS
                    cv2.putText(
                        final_frame,
                        f"FPS: {fps:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )
                    # Overlay Delay
                    cv2.putText(
                        final_frame,
                        f"Delay: {args.delay} ms",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )


                prev_delay = args.delay
                args.delay = cv2.getTrackbarPos('Delay (ms)', 'Delay Control')
                prev_alpha  = alpha
                alpha =  cv2.getTrackbarPos('TR %', 'Target Retention')

                if prev_alpha!=alpha:
                    
                    target_latent = create_source_latent(frame, args.face_attribute_direction, args.face_attribute_steps)
                    source_latent = blend(orig_source_latent, target_latent, alpha*0.01)



                buffer_end = time.time()
                buffer.append((final_frame, buffer_end))
                
                if (buffer_end-buffer[0][1])*1000>= args.delay:
                    if cam:
                        final_frame =  cv2.resize(buffer[0][0], None, fx=2.0, fy=2.0)
                        cam.send(final_frame)
                        cam.sleep_until_next_frame()
                        buffer.popleft()

                    else:
                        cv2.imshow('Live Face Swap', cv2.resize(buffer[0][0], None, fx=2.0, fy=2.0))
                        buffer.popleft()
                
            except Exception as e:
                print(f"Swap error: {e}")
                cv2.imshow('Live Face Swap', frame)
                traceback.print_exc()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):  # support both + and = on some keyboards
                args.delay += 50
                print(f"Delay increased to {args.delay} ms")
                cv2.setTrackbarPos('Delay (ms)', "Delay Control", args.delay)

            elif key == ord('-') or prev_delay>args.delay:
                if key==ord('-'):
                    args.delay = max(0, args.delay - 50)
                    cv2.setTrackbarPos('Delay (ms)', "Delay Control", args.delay)
                while (buffer_end - buffer[0][1])*1000>args.delay:
                    buffer.popleft()
                print(f"Delay decreased to {args.delay} ms")


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
