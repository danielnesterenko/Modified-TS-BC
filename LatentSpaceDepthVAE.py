import cv2
import torch
import pickle
import os
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from gym3.types import DictType
from distance_fns import DISTANCE_FUNCTIONS

import sys
sys.path.append('openai_vpt')

AGENT_RESOLUTION = (216, 384)

class LatentSpaceDepthVAE:
    def __init__(self, distance_fn='euclidean', device='cuda'):
        self.latents = []  # Python List while training, Numpy array while inference
        self.distance_function = DISTANCE_FUNCTIONS[distance_fn]
        self.device = device

    
    @torch.no_grad()
    def load(self, episode_actions, latents_folder='weights/ts_bc/latents_4x4x8_128/'):
        for vid_id, _ in episode_actions.episode_starts:
            _, name = vid_id.rsplit('/', 1)
            vid_latents = np.load(latents_folder + name + '.npy', allow_pickle=True)
            self.latents.append(vid_latents)

        self.latents = torch.from_numpy(np.vstack(self.latents)).to(self.device)
        self.latents = self.latents.squeeze(1)
        print(f'Loaded VAE_DepthAnything latent space with {len(self.latents)} latents from {latents_folder}')
        return self
    
    
    # Training done seperately in VAE/train_latent_space_vae.py

    '''old
    def get_distances(self, latent):

        first_list = self.latents[:, 0, :, :]
        first_list = first_list.squeeze(1)
        print(self.distance_function(first_list, latent[0]))

        return self.distance_function(first_list, latent[0])
    
    def get_distance(self, idx, latent):
        return self.distance_function(self.latents[idx][0], latent[0])
    '''

    def get_distances(self, latent):
        #distances = torch.norm(self.latents - latent, dim=1)
        return self.distance_function(self.latents, latent)
    
    def get_distance(self, idx, latent):
        return self.distance_function(self.latents[idx], latent.flatten().to(torch.float16))

    def get_nearest(self, latent): # TODO removed episode_starts
        # TODO assert latents is numpy array

        diffs = self.latents - latent
        diffs = abs(diffs).sum(1)  # Sum up along the single latents exponential difference to the current latent
        nearest_idx = diffs.argmin()#.to('cpu').item() # TODO remove .to('cpu').item()
        return nearest_idx

