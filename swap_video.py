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
import moviepy as mp
from moviepy import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from tqdm import tqdm
import shutil
from gfpgan import GFPGANer
import glob
from basicsr.utils import imwrite


# Setup face detector
faceAnalysis = FaceAnalysis(name='buffalo_l')
faceAnalysis.prepare(ctx_id=0, det_size=(512, 512))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Live face swap via webcam')
    parser.add_argument('--source', required=True, help='Path to source face image')
    parser.add_argument('--target_video', required=True, help='Path to target video')
    parser.add_argument('--modelPath', required=True, help='Path to the trained face swap model')
    parser.add_argument('--resolution', type=int, default=128, help='Resolution of the face crop')
    parser.add_argument('--face_attribute_direction', default=None, help='Path to face attribute direction.npy')
    parser.add_argument('--face_attribute_steps', type=float, default=0.0, help='Amount to move in attribute direction')
    parser.add_argument('--mouth_mask', action='store_true', help='Retain target mouth')
    parser.add_argument('-s', '--upscale', type=int, default=2, help='The final upsampling scale of the image. Default: 2')
    parser.add_argument('--bg_upsampler', type=str, default='realesrgan', help='background upsampler. Default: realesrgan')
    parser.add_argument('--bg_tile',type=int,default=400,help='Tile size for background sampler, 0 for no tile during testing. Default: 400')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
    parser.add_argument('--ext',type=str,default='auto',help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto')
    parser.add_argument('-w', '--weight', type=float, default=0.5, help='Adjustable weights.')
    parser.add_argument('-std', '--std', type=int, default=1, help='standard deviation for noise')
    parser.add_argument('-blur', '--blur', type=int, default=1, help='blur')





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


# @line_profiler.profile
def main():
    args = parse_arguments()

    if args.bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=args.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None

    restorer = GFPGANer(
        model_path="./models/GFPGANv1.3.pth",
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=bg_upsampler)



    video_path = args.target_video
    model = load_model(args.modelPath)
    video_forcheck = VideoFileClip(video_path)
    if video_forcheck.audio is None:
        no_audio = True
    else:
        no_audio = False

    del video_forcheck

    if not no_audio:
        video_audio_clip = AudioFileClip(video_path)

    video = cv2.VideoCapture(video_path)
    ret = True
    frame_index = 0

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    

    # video_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # video_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = video.get(cv2.CAP_PROP_FPS)
    temp_results_dir = "./temp_results"
    if not os.path.exists(temp_results_dir):
        os.mkdir(temp_results_dir)
    create_latent_flag = True

    for frame_index in tqdm(range(frame_count)): 
        ret, frame = video.read()
        if create_latent_flag:
            source = apply_color_transfer(source_path=args.source, target= frame)
            source_latent = create_source_latent(source, args.face_attribute_direction, args.face_attribute_steps)
            if source_latent is None:
                return "Face not found in source image"
            create_latent_flag = False
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
                if args.mouth_mask:
                    face_mask = Image.create_face_mask(target_face, frame)
                    _, mouth_cutout, mouth_box, lower_lip_polygon = (
                        Image.create_lower_mouth_mask(target_face, frame)
                    )
                    final_frame = Image.apply_mouth_area(
                        final_frame, mouth_cutout, mouth_box, face_mask, lower_lip_polygon
                    )
            cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), final_frame)
        except Exception as e:
            print(f"Swap error: {e}")



    
    img_list = sorted(glob.glob(os.path.join(temp_results_dir, '*')))
    if not os.path.exists("./temp_results2"):
        os.makedirs("./temp_results2", exist_ok=True)
    
    for img_path in img_list:
        # read image
        img_name = os.path.basename(img_path)
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # restore faces and background if necessary
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            has_aligned=args.aligned,
            only_center_face=args.only_center_face,
            paste_back=True,
            weight=args.weight)

        # save restored img
        if restored_img is not None:
            if args.ext == 'auto':
                extension = ext[1:]
            else:
                extension = args.ext
            if args.suffix is not None:
                save_restore_path = os.path.join("./temp_results2", 'restored_imgs', f'{basename}_{args.suffix}.{extension}')
            else:
                save_restore_path = os.path.join("./temp_results2", 'restored_imgs', f'{basename}.{extension}')


            # Generate subtle Gaussian noise
            mean = 0
            std_dev = args.std  # reduce this value for less noise (try 0.1 to 0.5)
            noise = np.random.normal(mean, std_dev, restored_img.shape).astype(np.float32)
            # Add noise
            textured_img = restored_img + noise
            # Clip and convert to uint8
            textured_img = np.clip(textured_img, 0, 255).astype(np.uint8)
            textured_blurred_img = cv2.GaussianBlur(textured_img, (args.blur, args.blur), 0)

            # std_dev = 20  # reduce this value for less noise (try 0.1 to 0.5)
            # textured_blurred_img = cv2.GaussianBlur(textured_img, (31, 31), 0)

            imwrite(textured_blurred_img, save_restore_path)




    # Get all filenames in the directory
    all_files = os.listdir("./temp_results2")

    # Filter for files ending in .jpg
    image_filenames = [os.path.join("./temp_results2", f) for f in all_files if f.lower().endswith('.jpg')]

    clips = ImageSequenceClip(image_filenames, fps = fps)
    if not no_audio:
        clips = clips.with_audio(video_audio_clip)
    clips.write_videofile("output.mp4", codec='libx264', audio_codec='aac')

    shutil.rmtree("./temp_results2")
    shutil.rmtree(temp_results_dir)






if __name__ == "__main__":
    main()







        #  # restore faces and background if necessary
        # cropped_faces, restored_faces, restored_img = restorer.enhance(
        #     frame,
        #     has_aligned=args.aligned,
        #     only_center_face=args.only_center_face,
        #     paste_back=True,
        #     weight=args.weight)

        # # save restored img
        # if restored_img is not None:
        #     if args.ext == 'auto':
        #         extension = ext[1:]
        #     else:
        #         extension = args.ext
        #     if args.suffix is not None:
        #         save_restore_path = os.path.join(temp_results_dir, 'restored_imgs', f'{basename}_{args.suffix}.{extension}')
        #     else:
        #         save_restore_path = os.path.join(temp_results_dir, 'restored_imgs', f'{basename}.{extension}')



        #     mean = 0
        #     std_dev = args.std  
        #     noise = np.random.normal(mean, std_dev, restored_img.shape).astype(np.float32)
        #     # Add noise
        #     textured_img = restored_img + noise

        #     textured_img = np.clip(textured_img, 0, 255).astype(np.uint8)
        #     textured_blurred_img = cv2.GaussianBlur(textured_img, (args.blur, args.blur), 0)

        #     # std_dev = 20  # reduce this value for less noise (try 0.1 to 0.5)
        #     # textured_blurred_img = cv2.GaussianBlur(textured_img, (31, 31), 0)
