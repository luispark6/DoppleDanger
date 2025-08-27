

from datetime import datetime
import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F

from discrimintator_training.Discriminatorv3_3 import iresnet50
import Image
import ModelFormat
from StyleTransferLoss import StyleTransferLoss
import onnxruntime as rt

import cv2
from insightface.data import get_image as ins_get_image
from insightface.app import FaceAnalysis
import face_align

from StyleTransferModel_128 import StyleTransferModel
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pickle
import matplotlib.pyplot as plt



inswapper_128_path = './models/inswapper_128.onnx'
img_size = 128

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

inswapperInferenceSession = rt.InferenceSession(inswapper_128_path, providers=providers)

faceAnalysis = FaceAnalysis(name='buffalo_l')
faceAnalysis.prepare(ctx_id=0, det_size=(512, 512))




def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
style_loss_fn = StyleTransferLoss().to(get_device())



def createFakeImage(datasetDir, image, enableDataAugmentation, steps, resolution, device):
    targetFaceIndex = random.randint(0, len(image)-1)
    sourceFaceIndex = random.randint(0, len(image)-1)

    target_img=cv2.imread(f"{datasetDir}/{image[targetFaceIndex]}")
    if enableDataAugmentation and steps % 2 == 0:
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR)
    faces = faceAnalysis.get(target_img)

    if targetFaceIndex != sourceFaceIndex:
        source_img = cv2.imread(f"{datasetDir}/{image[sourceFaceIndex]}")
        faces2 = faceAnalysis.get(source_img)
    else:
        faces2 = faces

    if len(faces) > 0 and len(faces2) > 0:
        new_aligned_face, _ = face_align.norm_crop2(target_img, faces[0].kps, img_size)
        blob = Image.getBlob(new_aligned_face)
        latent = Image.getLatent(faces2[0])
    else:
        return createFakeImage(datasetDir, image, enableDataAugmentation, steps, resolution, device)

    if targetFaceIndex != sourceFaceIndex:
        input = {inswapperInferenceSession.get_inputs()[0].name: blob,
                inswapperInferenceSession.get_inputs()[1].name: latent}

        expected_output = inswapperInferenceSession.run([inswapperInferenceSession.get_outputs()[0].name], input)[0]
    else:
        expected_output = blob

    expected_output_tensor = torch.from_numpy(expected_output).to(device)

    if resolution != 128:
        new_aligned_face, _ = face_align.norm_crop2(target_img, faces[0].kps, resolution)
        blob = Image.getBlob(new_aligned_face, (resolution, resolution))

    latent_tensor = torch.from_numpy(latent).to(device)
    target_input_tensor = torch.from_numpy(blob).to(device)

    return target_input_tensor, latent_tensor, expected_output_tensor

def add_instance_noise(images, std=0.1):
    noise = torch.randn_like(images) * std
    return images + noise

def train(datasetDir, learning_rate=0.0001, model_path=None, d_model_path=None,  outputModelFolder='', saveModelEachSteps = 1, stopAtSteps=None, logDir=None, previewDir=None, saveAs_onnx = False, resolutions = [128], enableDataAugmentation = False):
    device = get_device()
    print(f"Using device: {device}")
    train_g = True 
    train_d = True

    model = StyleTransferModel().to(device)
    discriminator = iresnet50().to(device)  # Add discriminator
 
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00001)  # S

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  

    if model_path is not None or d_model_path is not None:

        if model_path:
            # model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            # print(f"Loaded model from {model_path}")
            # lastSteps = int(model_path.split('-')[-1].split('.')[0])
            # print(f"Resuming training from step {lastSteps}")


            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            
            lastSteps = checkpoint.get('steps', 0)
            

        if d_model_path:

            d_checkpoint = torch.load(d_model_path, map_location=device)
            discriminator.load_state_dict(d_checkpoint['model_state_dict'], strict=False)
            optimizer_D.load_state_dict(d_checkpoint['optimizer_state_dict'])
           
        # if model_path and d_model_path :
        #     print("WARNING: Non-matching steps between generator and discriminator")
        #     return
    
      

        print(f"Loaded generator model from {model_path}")
        print(f"Loaded discriminator model from {d_model_path}")

        print(f"Resuming training from step {lastSteps}")
    else:
        lastSteps = 0

    model.train()
    model = model.to(device)
    discriminator.train()
   
    
    # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # Initialize TensorBoard writer
    if logDir is not None:
        train_writer = SummaryWriter(os.path.join(logDir, "training"))
        val_writer = SummaryWriter(os.path.join(logDir, "validation"))

    steps = 0
    image = os.listdir(datasetDir)
    resolutionIndex = 0

    batch_size = 5
    warmup_steps= 200
    # Training loop
    while True:
        start_time = datetime.now()
        
        resolution = resolutions[resolutionIndex%len(resolutions)]
        optimizer.zero_grad()

        # if steps % 100 == 0 or True:
        real_images_list = []
        fake_images_list = []

        if steps < warmup_steps and not d_model_path:
            # Train only discriminator
            train_d = True
            train_g = False
        else:
            # Normal joint training
            train_d = True
            train_g = True

        while len(real_images_list)!=batch_size:
            realFaceIndex = random.randint(0, len(image)-1)
            real_img = cv2.imread(f"{datasetDir}/{image[realFaceIndex]}")
            faces3 = faceAnalysis.get(real_img)
            if len(faces3) == 0 : continue

            aligned_real_face, _ = face_align.norm_crop2(real_img, faces3[0].kps, resolution)
            real_images = torch.from_numpy(Image.getBlob(aligned_real_face, (resolution, resolution))).to(device)

            real_images = F.interpolate(real_images, size=(224, 224), mode='bilinear', align_corners=False)
            real_images_list.append(real_images)

        while len(fake_images_list)!=batch_size:
            target_input_tensor, latent_tensor, expected_output_tensor = createFakeImage(datasetDir, image, enableDataAugmentation, steps, resolution, device)

            with torch.no_grad():
                output = model(target_input_tensor, latent_tensor)

            fake_images = output.detach()  # Detach to avoid backprop through generator
            fake_images = F.interpolate(fake_images, size=(224, 224), mode='bilinear', align_corners=False)

            fake_images_list.append(fake_images)

        if train_d :
            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Use ground truth as real samples
            fake_images_list = torch.stack(fake_images_list, 1).to(device)
            real_images_list = torch.stack(real_images_list, 1).to(device)



            

           
            real_pred = discriminator(real_images_list[0])
            fake_pred = discriminator(fake_images_list[0])
            real_labels = torch.ones_like(real_pred)   
            fake_labels = torch.zeros_like(fake_pred)      

            d_loss_real = F.binary_cross_entropy_with_logits(real_pred, real_labels)
            d_loss_fake = F.binary_cross_entropy_with_logits(fake_pred, fake_labels)

    
            print("Real pred mean:", torch.sigmoid(real_pred).mean().item())
            print("Fake pred mean:", torch.sigmoid(fake_pred).mean().item())

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

  

        #Train Gen
        if train_g:
          
            target_input_tensor, latent_tensor, expected_output_tensor = createFakeImage(datasetDir, image, enableDataAugmentation, steps, resolution, device)

            output = model(target_input_tensor, latent_tensor)

            if (resolution != 128):
                output = F.interpolate(output, size=(128, 128), mode='bilinear', align_corners=False)

            content_loss, identity_loss = style_loss_fn(output, expected_output_tensor)
            
            output_224 = F.interpolate(output, size=(224, 224), mode='bilinear', align_corners=False)

            fake_pred = discriminator(output_224)



    
            adversarial_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))


            loss = content_loss + adversarial_loss 

            if identity_loss is not None:
                loss +=identity_loss
                
            
            loss.backward()

            optimizer.step()
        

        steps+=1
        totalSteps = steps + lastSteps


        

        if logDir is not None:
            if train_g:
                train_writer.add_scalar("Loss/total", loss.item(), totalSteps)
                train_writer.add_scalar("Loss/content_loss", content_loss.item(), totalSteps)
                train_writer.add_scalar("Loss/adversarial_loss", adversarial_loss.item(), totalSteps)
                if identity_loss is not None:
                    train_writer.add_scalar("Loss/identity_loss", identity_loss.item(), totalSteps)
            if train_d:
                train_writer.add_scalar("Loss/d_loss", d_loss.item(), totalSteps)
                train_writer.add_scalar("Loss/d_loss_fake", d_loss_fake.item(), totalSteps)
                train_writer.add_scalar("Loss/d_loss_real", d_loss_real.item(), totalSteps)

        elapsed_time = datetime.now() - start_time

        if train_d:
            print(f"Total Steps: {totalSteps}, Step: {steps}, D_Loss: {d_loss.item():.4f}, d_loss_real: {d_loss_real.item():.4f}, d_loss_fake: {d_loss_fake.item():.4f},  Elapsed time: {elapsed_time}")
        if train_g:
            print(f"Total Steps: {totalSteps}, Step: {steps}, G_Loss: {loss.item():.4f}, Identity_Loss: {identity_loss}, Content_Loss: {content_loss} Elapsed time: {elapsed_time}")

        if steps % saveModelEachSteps == 0:
            if train_g:
                outputModelPath = f"reswapper-{totalSteps}.pth"
                if outputModelFolder != '':
                    outputModelPath = f"{outputModelFolder}/{outputModelPath}"
                saveModel(model, optimizer, outputModelPath, totalSteps)
            if train_d:
                discriminatorModelPath = f"discriminator-{totalSteps}.pth"
                if outputModelFolder != '':
                    discriminatorModelPath = f"{outputModelFolder}/{discriminatorModelPath}"
                saveModel(discriminator, optimizer_D, discriminatorModelPath, totalSteps)

            if train_g:
                validation_total_loss, validation_content_loss, validation_identity_loss, swapped_face = validate(outputModelPath)
                if previewDir is not None:
                    cv2.imwrite(f"{previewDir}/{totalSteps}.jpg", swapped_face)

                if logDir is not None:
                    val_writer.add_scalar("Loss/total", validation_total_loss.item(), totalSteps)
                    val_writer.add_scalar("Loss/content_loss", validation_content_loss.item(), totalSteps)
                    if validation_identity_loss is not None:
                        val_writer.add_scalar("Loss/identity_loss", validation_identity_loss.item(), totalSteps)

            if saveAs_onnx :
                ModelFormat.save_as_onnx_model(outputModelPath)
        

        if not steps%saveModelEachSteps:
            if steps!= saveModelEachSteps:
                os.remove(f"{outputModelFolder}/reswapper-{totalSteps-saveModelEachSteps}.pth")
                os.remove(f"{outputModelFolder}/discriminator-{totalSteps-saveModelEachSteps}.pth")


            
        

        if stopAtSteps is not None and steps == stopAtSteps:
            exit()

        resolutionIndex += 1


def saveModel(model, optimizer, outputModelPath, steps):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'steps': steps
    }, outputModelPath)

def load_model(model_path):
    device = get_device()
    model = StyleTransferModel().to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model.eval()
    return model

def swap_face(model, target_face, source_face_latent):
    device = get_device()

    target_tensor = torch.from_numpy(target_face).to(device)
    source_tensor = torch.from_numpy(source_face_latent).to(device)

    with torch.no_grad():
        swapped_tensor = model(target_tensor, source_tensor)

    swapped_face = Image.postprocess_face(swapped_tensor)
    
    return swapped_face, swapped_tensor

# test image

test_img = cv2.imread("../elonmusk.png")
test_faces = faceAnalysis.get(test_img)
target_face = test_faces[0]

# x1,y1,x2,y2 = target_face['bbox']
aligned_face, M = face_align.norm_crop2(test_img, target_face.kps, img_size)
test_target_face = Image.getBlob(aligned_face)



test_img2 = cv2.imread("../American-actor-Leonardo-DiCaprio-2016.png")
test_faces2 = faceAnalysis.get(test_img2)
target_face2 = test_faces2[0]

test_l = Image.getLatent(target_face2)



test_input = {inswapperInferenceSession.get_inputs()[0].name: test_target_face,
        inswapperInferenceSession.get_inputs()[1].name: test_l}

test_inswapperOutput = inswapperInferenceSession.run([inswapperInferenceSession.get_outputs()[0].name], test_input)[0]


expected_output = np.clip(test_inswapperOutput[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
expected_output=cv2.cvtColor(expected_output, cv2.COLOR_RGB2BGR)

cv2.imwrite(f"./expected_output.jpg", expected_output)

def validate(modelPath):
    model = load_model(modelPath)
    swapped_face, swapped_tensor= swap_face(model, test_target_face, test_l)


    validation_content_loss, validation_identity_loss = style_loss_fn(swapped_tensor, torch.from_numpy(test_inswapperOutput).to(get_device()))

    validation_total_loss = validation_content_loss
    if validation_identity_loss is not None:
        validation_total_loss += validation_identity_loss

    return validation_total_loss, validation_content_loss, validation_identity_loss, swapped_face

def main():
    outputModelFolder = "train_model"
    modelPath = None
    modelPath = f"train_model_keep/reswapper-1592500.pth"

    logDir = "training/log"
    previewDir = "training/preview"
    datasetDir = "../ffhq"

    os.makedirs(outputModelFolder, exist_ok=True)
    os.makedirs(previewDir, exist_ok=True)

    train(
        datasetDir=datasetDir,
        model_path=modelPath,
        d_model_path = "train_model_keep/discriminator-1592500.pth",
        learning_rate=0.000001,
        resolutions = [128],
        enableDataAugmentation=False,
        outputModelFolder=outputModelFolder,
        saveModelEachSteps = 5000,
        stopAtSteps = 20000,
        logDir=f"{logDir}/{datetime.now().strftime('%Y%m%d %H%M%S')}",
        previewDir=previewDir)
                    
if __name__ == "__main__":
    main()
