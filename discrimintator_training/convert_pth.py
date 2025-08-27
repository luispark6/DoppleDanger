from datetime import datetime
import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F

import Image
import ModelFormat
from StyleTransferLoss import StyleTransferLoss
import onnxruntime as rt

import cv2
from insightface.data import get_image as ins_get_image
from insightface.app import FaceAnalysis
import face_align

from StyleTransferModel_128 import StyleTransferModel


import numpy as np
import matplotlib.pyplot as plt
import pickle



model = StyleTransferModel().to('cuda')

checkpoint = torch.load("./train_model_keep/reswapper-1597500.pth", map_location="cuda")
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

torch.save(model.state_dict(), "./models/reswapper-1597500-dis.pth")
