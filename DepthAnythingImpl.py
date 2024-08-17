import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from depth_anything.transforms import Compose
from tqdm import tqdm
import sys

from depth_anything.dpt import DepthAnything
from depth_anything.dpt_original import DepthAnythingOG
from depth_anything.dpt_latent_space import DepthAnythingSpace
from depth_anything.dpt_decode import DepthAnythingDecoder
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

os.environ['TRANSFORMERS_CACHE'] = '/media/local/DanielNesterenko/thesis/'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

class DepthAnythingImpl:
    def __init__(self, device='cuda'):
        self.device = device
        self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format('vitl'), cache_dir='/media/local/DanielNesterenko/thesis/').to(device).eval()
        self.depth_anything_original = DepthAnythingOG.from_pretrained('LiheYoung/depth_anything_{}14'.format('vitl'), cache_dir='/media/local/DanielNesterenko/thesis/').to(device).eval()
        self.depth_anything_latent_space = DepthAnythingSpace.from_pretrained('LiheYoung/depth_anything_{}14'.format('vitl'), cache_dir='/media/local/DanielNesterenko/thesis/').to(device).eval()
        self.depth_anything_decode = DepthAnythingDecoder.from_pretrained('LiheYoung/depth_anything_{}14'.format('vitl'), cache_dir='/media/local/DanielNesterenko/thesis/').to(device).eval()
        self.transform = transform


    def depthAnything_rgb_to_depth_frame(self, frame):

        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) / 255.0

        h, w = image.shape[:2]
        image = self.transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = self.depth_anything_original(image)

        prediction = F.interpolate(prediction[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min()) * 255.0


        output = prediction.cpu().numpy().astype(np.uint8)

        #output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
        output = cv2.applyColorMap(output, cv2.COLORMAP_INFERNO)
        return output
    

    def depthAnything_rgb_to_depth_frame_greyscaled(self, frame):

        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) / 255.0

        h, w = image.shape[:2]
        image = self.transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = self.depth_anything_original(image)

        prediction = F.interpolate(prediction[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min()) * 255.0


        output = prediction.cpu().numpy().astype(np.uint8)
        output = np.repeat(output[..., np.newaxis], 3, axis=-1)

        return output
    

    def depthAnything_latent_by_frame(self, frame):

        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) / 255.0

        h, w = image.shape[:2]
        image = self.transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = self.depth_anything_latent_space(image)

        return prediction