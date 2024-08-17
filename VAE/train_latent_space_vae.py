import torch
from tqdm import tqdm
import numpy as np
from torch import nn
from model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import sys
import cv2
from VAE2 import VAE

import sys
sys.path.append('../')  # Add the parent directory to the Python path
from VPTDatasetDepthAnything2 import VPTDatasetDepthAnything

#Configuration
AGENT_RESOLUTION = (216, 384)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256 
LR_RATE = 3e-4 # Karpathy constant
SLIDING_WINDOW_SIZE = 16

# Dataset
dataset = VPTDatasetDepthAnything()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((216, 384))  # Resize to the desired dimensions
])

# Model
model = VAE().to(DEVICE)
weights = torch.load("./vae_training/vae_model_y.pth")
model.load_state_dict(weights)
model.eval()


def main():
    for i in range(20): #adjust len
        if i >= len(dataset):
            break
        frames, _, vid_id = dataset[i]
        frames = preprocess(frames)
        train_episode_depth_vae(frames, vid_id)


def train_episode_depth_vae(frames, vid_id):
    episode_latents = []

    # Preparing Batches
    for i in range(SLIDING_WINDOW_SIZE-1, len(frames), batch_size):
        batch = frames[i:i+batch_size] # Each batch containing batch_size amount of frames but neglecting first 16 frames due to MineClip architecture

        print("####################5")
        print(batch)
        print(batch.shape)
        sys.exit()
        
        latent_batch = encode_frame(batch)
        print('##################4')
        print(latent_batch.shape)
        sys.exit()
        for latent_frame in range(latent_batch.size(0)):

            latent = latent_batch[latent_frame]
            latent = latent.cpu().numpy().astype('float16').flatten()
            episode_latents.append(latent)
        
    print(f'###########TESTING {len(episode_latents)}')
        
    del(frames)
    print((vid_id.rsplit('/', 1)[-1]))
    np.save('latents/' + vid_id.rsplit('/', 1)[-1], np.array(episode_latents))     

    #TODO's: Latents fertig -> Jetzt LatentSpace bauen -> Agent daruf hooken (obs muss wahrscheinlich 256 dupliziert werden, damit processed in VAE), Evaluieren.


@torch.no_grad()
def encode_frame(batch):
    batch = batch.to(DEVICE)
    #print(f'BEGINNING {batch.shape}')
    #print(f'BATCH: {batch.shape}')
    _, mu, logvar = model(batch)
    latent_batch = model.reparameterize(mu, logvar)
    #print(f'LATENT: {latent_batch.shape}')
    return latent_batch


def preprocess(frames):

    print("###################1")
    print(frames)
    print(frames.shape)


    frames_ret = []
    for frame in frames:
        #print("###################3")
        #print(frame.shape)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
        frame_tensor = transform(frame)
        frames_ret.append(frame_tensor)
    video_tensor = torch.stack(frames_ret)

    print('################2')
    print(video_tensor)
    print(video_tensor.shape)
    #sys.exit()

    return video_tensor



if __name__ == "__main__":
    main()


