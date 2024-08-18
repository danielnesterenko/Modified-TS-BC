import os
import argparse
import sys
from tqdm import tqdm

from VAE.build_latent_space_vae import build_latent_space
from VPTDataset import VPTDataset
from VPTDatasetDepthAnything import VPTDatasetDepthAnything
from EpisodeActions import EpisodeActions
from LatentSpaceMineCLIP_Patches import LatentSpaceMineCLIP, load_mineclip
from LatentSpaceDepthVAE import LatentSpaceDepthVAE

def main(args):
    print('Computing Latent Vectors...')

    rgb_dataset = VPTDataset()
    depth_dataset = VPTDatasetDepthAnything()

    episode_actions = EpisodeActions()
    latent_space_mineclip = LatentSpaceMineCLIP()
    mineclip_model = load_mineclip()

    iterator = range(args.batch_size) if args.random_sample_size is None else tqdm(range(args.random_sample_size))

    for i in tqdm(iterator):
        if args.random_sample_size is None:
            idx = args.batch_idx * args.batch_size + i
            if idx >= len(rgb_dataset):
                break
            frames, actions, vid_id = rgb_dataset[idx]
            depth_frames, _, depth_vid_id = depth_dataset[idx]
        else:
            frames, actions, vid_id = rgb_dataset.get_random()

        if frames is None:
            print(f'SKIPPING index: {idx}')
            continue

        episode_actions.train_episode(actions, vid_id)
        build_latent_space(depth_frames, depth_vid_id) # Building VAE embedding space
        latent_space_mineclip.train_episode(mineclip_model, frames, vid_id) # Building MineCLIP + Patch embeddings embedding space
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--random-sample-size', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--batch-idx', type=int, default=0)
    args = parser.parse_args()

    main(args)
