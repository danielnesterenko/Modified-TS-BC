import torch
from tqdm import tqdm
import numpy as np
from torch import nn
from model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import os
from PIL import Image
import sys
import cv2
#from VAE2 import VAE
from z_vanilla_vae import VanillaVAE

# UNCOMMENT IF TRAINING OF NEED DATA
import sys
sys.path.append('../')  # Add the parent directory to the Python path
from VPTDatasetDepthAnything2 import VPTDatasetDepthAnything

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
weights = torch.load("/media/local/DanielNesterenko/thesis/TS-BC/VAE/trained_models/model_vid10000_2_4x4x8.pth")
model.load_state_dict(weights)
model.eval()


def main():
    for i in range(20): #adjust len
        if i >= len(dataset):
            break
        frames, _, vid_id = dataset[i]

        build_latent_space(frames, vid_id)
        #frame = preprocess_img(frames[0])
        #encoded_frame = model.encode(frame)

        #encoded_decoded_output = model.forward(frame)

        #train_episode_depth_vae(frames, vid_id)


def build_latent_space(frames, vid_id):
    episode_latents = []
    for i in range(SLIDING_WINDOW_SIZE-1, len(frames)):
        frame = preprocess_img(frames[i])

        # cv2 altered depth channels to 3 (almost similar) channels, therefore take one of the channels and make it R=G=B
        depth_channel = frame[:, 1:2, :, :]
        frame = depth_channel.repeat(1, 3, 1, 1)

        latent = encode_frame(frame)

        #latent[0] = latent[0].detach().numpy()
        #latent[1] = latent[1].detach().numpy()
        latent = latent.detach().numpy()

        #sys.exit()
        #latent = latent.cpu().numpy().astype('float16').flatten()
        episode_latents.append(latent)

    #print(episode_latents[0][1])

    print(f'###########TESTING {len(episode_latents)}')

    '''
    z = model.reparameterize(torch.tensor(episode_latents[0][0]), torch.tensor(episode_latents[0][1]))
    decoded = [model.decode(z), input, torch.tensor(episode_latents[0][0]), torch.tensor(episode_latents[0][1])]
    vutils.save_image(decoded[0],
                          os.path.join(f"./ZZZTEST.png"),
                          normalize=True)
    '''
        
    del(frames)
    print((vid_id.rsplit('/', 1)[-1]))
    #episode_latents = episode_latents.detach().numpy()
    np.save('./latents_4x4x8_128/' + vid_id.rsplit('/', 1)[-1], np.array(episode_latents))


@torch.no_grad()
def encode_frame(frame):
    encoded = model.encode(frame)
    return encoded


def preprocess_img(frame):
    img = Image.fromarray(frame)
    transformed_img = transform(img)
    return torch.unsqueeze(transformed_img, 0)


if __name__ == "__main__":
    main()


