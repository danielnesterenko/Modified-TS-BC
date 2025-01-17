import cv2
import numpy as np
import argparse
import skvideo.io
import os

from distance_fns import DISTANCE_FUNCTIONS
from tqdm import tqdm
from run_agent_rank_analysis_patches import main as run_agent_main


# Specify goals and related seeds: Each goal will be ran on every seed
goals = ['NO_GOAL']
seeds = [164]#, 456, 789]


def main(args):

    os.makedirs(f'{args.output_dir}/', exist_ok=True)

    for goal in range(len(goals)):
        for seed in range(len(seeds)):
        
            args.goal = goals[goal]
            args.seed = seeds[seed]

            # run agent with current goal and seed -> creates frame comparisons after run.
            run_agent_main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--distance-fn', type=str, default='cosine', choices=DISTANCE_FUNCTIONS.keys())

    # neccesary args for agent
    parser.add_argument('--title', type=str, default='recording')
    parser.add_argument('--search-folder', type=str, default='weights/ts_bc/')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max-frames', type=int, default=1*1*20)
    parser.add_argument('--goal', type=str, default='gather wood')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    main(args)