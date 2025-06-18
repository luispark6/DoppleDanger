
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import line_profiler
import torch

emap = np.load("emap.npy")
input_std = 255.0
input_mean = 0.0

def postprocess_face(face_tensor):
    face_tensor = face_tensor.squeeze().cpu().detach()
    face_np = (face_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

    return face_np

def getBlob(aimg, input_size = (128, 128)):
    blob = cv2.dnn.blobFromImage(aimg, 1.0 / input_std, input_size,
                            (input_mean, input_mean, input_mean), swapRB=True)
    return blob

def getLatent(source_face):
    latent = source_face.normed_embedding.reshape((1,-1))
    latent = np.dot(latent, emap)
    latent /= np.linalg.norm(latent)

    return latent

def blend_swapped_image(swapped_face, target_image, M):
    # get image size
    h, w = target_image.shape[:2]
    
    # create inverse affine transform
    M_inv = cv2.invertAffineTransform(M)
    
    # warp swapped face back to target space
    warped_face = cv2.warpAffine(
        swapped_face,
        M_inv,
        (w, h),
        borderValue=0.0
    )
    
    # create initial white mask
    img_white = np.full(
        (swapped_face.shape[0], swapped_face.shape[1]),
        255,
        dtype=np.float32
    )
    
    # warp white mask to target space
    img_mask = cv2.warpAffine(
        img_white,
        M_inv,
        (w, h),
        borderValue=0.0
    )
    
    # threshold and refine mask
    img_mask[img_mask > 20] = 255
    
    # calculate mask size for kernel scaling
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    if len(mask_h_inds) > 0 and len(mask_w_inds) > 0:  # safety check
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h * mask_w))
        
        # erode mask
        k = max(mask_size // 10, 10)
        kernel = np.ones((k, k), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        
        # blur mask
        k = max(mask_size // 20, 5)
        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
    
    # normalize mask
    img_mask = img_mask / 255.0
    img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
    
    # blend images using mask
    result = img_mask * warped_face + (1 - img_mask) * target_image.astype(np.float32)
    result = result.astype(np.uint8)
    
    return result

def drawKeypoints(image, keypoints, colorBGR, keypointsRadius=2):
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(image, (x, y), radius=keypointsRadius, color=colorBGR, thickness=-1) # BGR format, -1 means filled circle

@line_profiler.profile
def blend_swapped_image_gpu(swapped_face, target_image, M, minimal):
    h, w = target_image.shape[:2]
    M_inv = cv2.invertAffineTransform(M)

    # Warp swapped face
    warped_face = cv2.warpAffine(
        swapped_face,
        M_inv,
        (w, h),
        borderMode= 0
    )
    # Create white mask
    img_white = np.full(swapped_face.shape[:2], 255, dtype=np.uint8)
    img_mask = cv2.warpAffine(img_white, M_inv, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    # Threshold and refine mask

    _, _, w_box, h_box = cv2.boundingRect(img_mask)
    mask_size = int(np.sqrt(w_box * h_box))


    k = max(mask_size // 10, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    img_mask = cv2.erode(img_mask, kernel, iterations=1)


    k_blur = max(mask_size // 20, 5)
    blur_size = (2 * k_blur + 1, 2 * k_blur + 1)
    img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)

    # Move to GPU for blending
    img_mask = torch.from_numpy(img_mask).to('cuda').unsqueeze(2)/255  # HWC, single channel
    warped_face = torch.from_numpy(warped_face).to(device='cuda')
    target_image = torch.from_numpy(target_image).to(device='cuda')
    # Blend
    result = img_mask * warped_face + (1 - img_mask) * target_image
    result = result.clamp(0, 255).byte().cpu().numpy()  # Back to CPU and uint8

    return result