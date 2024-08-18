import torch
from tqdm import tqdm
import numpy as np
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import os
from PIL import Image
import sys
import cv2
from VAE.vanilla_vae import VanillaVAE

import sys
sys.path.append('../')  # Add the parent directory to the Python path
from VPTDatasetDepthAnything import VPTDatasetDepthAnything

#Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
patch_size = 16
crop_size = 640
in_channels = 3
latent_dim = 512
SLIDING_WINDOW_SIZE = 16

# Dataset
dataset = VPTDatasetDepthAnything()
transform = transforms.Compose([transforms.CenterCrop(crop_size),
                                            transforms.Resize(patch_size),
                                            transforms.ToTensor()])

# Model
model = VanillaVAE(in_channels, latent_dim)
weights = torch.load("weights/pretrained_vae/model_vid10000_2_4x4x8.pth")
model.load_state_dict(weights)
model.eval()

def build_latent_space(frames, vid_id):
    episode_latents = []
    for i in range(SLIDING_WINDOW_SIZE-1, len(frames)):
        frame = preprocess_img(frames[i])

        # cv2 altered depth channels to 3 (almost similar) channels, therefore take one of the channels and make it R=G=B
        depth_channel = frame[:, 1:2, :, :]
        frame = depth_channel.repeat(1, 3, 1, 1)

        latent = encode_frame(frame)
        latent = latent.detach().numpy()

        episode_latents.append(latent)
        
    del(frames)
    print((vid_id.rsplit('/', 1)[-1]))
    sys.path.append('../')
    np.save('weights/ts_bc/latents_4x4x8_128/' + vid_id.rsplit('/', 1)[-1], np.array(episode_latents))
    print(f'### Encoded {len(episode_latents)} VAE embeddings.')
    print(f'### VAE embeddings finished for video: {vid_id}.')


@torch.no_grad()
def encode_frame(frame):
    encoded = model.encode(frame)
    return encoded


def preprocess_img(frame):
    img = Image.fromarray(frame)
    transformed_img = transform(img)
    return torch.unsqueeze(transformed_img, 0)



