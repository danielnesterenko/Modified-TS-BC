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

from VPTDatasetDepthAnything import VPTDatasetDepthAnything

#Configuration
AGENT_RESOLUTION = (320, 180)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 180
H_DIM = 960 # may adjust
Z_DIM = 20 # may adjust
NUM_EPOCHS = 20
BATCH_SIZE = 32
LR_RATE = 3e-4 # Karpathy constant

# Dataset
dataset = VPTDatasetDepthAnything()
transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(AGENT_RESOLUTION),
            transforms.ToTensor()])

# Model
model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.MSELoss(reduction="mean")


def main():
    for i in range(2): #adjust len
        if i >= len(dataset):
            break
        frames, _, vid_id = dataset[i]
        frames = resize(frames)

        for frame in tqdm(frames):
            # Forward
            x_reconstructed, mu, sigma = model(frame)

            # Compute loss
            reconstruction_loss = loss_fn(x_reconstructed, frame)
            kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            # Backpropagation
            loss = reconstruction_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def resize(frames):
    resized_frames = np.empty((frames.shape[0], AGENT_RESOLUTION[1], AGENT_RESOLUTION[0], 3), dtype=np.uint8)
    for ts in range(frames.shape[0]):
        resized_frame = cv2.resize(frames[ts], AGENT_RESOLUTION)
        resized_frames[ts] = resized_frame
    frames = [transform(frame).unsqueeze(0).to(DEVICE) for frame in resized_frames]
    return frames



if __name__ == "__main__":
    main()


