import os
import cv2
import torch
import minerl
import argparse
import sys
from tqdm import tqdm
from DepthAnythingImpl import DepthAnythingImpl
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

from programmatic_eval import ProgrammaticEvaluator
from distance_fns import DISTANCE_FUNCTIONS
from ModifiedTSBC import TargetedSearchAgent

ENV_KWARGS = dict(
    fov_range=[70, 70],
    frameskip=1,
    gamma_range=[2, 2],
    guiscale_range=[1, 1],
    resolution=[640, 360],
    cursor_size_range=[16.0, 16.0],
)
CLIP_RESOLUTION = (256, 160)

def main(args):
    os.makedirs(f'{args.output_dir}/', exist_ok=True)
    os.makedirs(f'{args.output_dir}/agent_recordings/', exist_ok=True)
    os.makedirs(f'{args.output_dir}/programmatic_results/', exist_ok=True)
    os.makedirs(f'{args.output_dir}/agent_logs/', exist_ok=True)
    os.makedirs(f'{args.output_dir}/agent_diff_logs/', exist_ok=True)

    depth_anything = DepthAnythingImpl()

    video_writer = cv2.VideoWriter(f'{args.output_dir}/agent_recordings/{args.title}_{str(args.run)}_seed{args.seed}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 360))

    # HumanSurvival is a sandbox `gym` environment for Minecraft with no set goal or timeframe
    env = HumanSurvival(**ENV_KWARGS).make()
    if args.seed is not None:
        env.seed(args.seed)

    agent = TargetedSearchAgent(env, search_folder=args.search_folder, distance_fn=args.distance_fn, device=args.device)
    agent.set_goal(args.goal)

    obs = env.reset()
    prog_evaluator = ProgrammaticEvaluator(obs)

    print('### Starting agent')
    with torch.no_grad():
        # transform first frame to depth
        depth_frame = depth_anything.depthAnything_rgb_to_depth_frame_greyscaled(obs['pov'])
        clip_window_frames = []
        clip_tensor = None
        clip_search_enabled = False
        for frame in tqdm(range(args.max_frames)):

            # gathering initial frames for window
            if frame <= 15:
                clip_format_frame = cv2.resize(obs['pov'], CLIP_RESOLUTION)
                clip_window_frames.append(clip_format_frame)
            # building tensor for MineCLIP
            elif frame == 16:
                clip_search_enabled = True
                clip_tensor = torch.tensor(clip_window_frames).unsqueeze(0).permute(0, 1, 4, 3, 2)
                clip_format_frame = cv2.resize(obs['pov'], CLIP_RESOLUTION)
                clip_tensor = increment_clip_window(clip_tensor, clip_format_frame)
            # remove first frame, add new frame as last
            else:
                clip_format_frame = cv2.resize(obs['pov'], CLIP_RESOLUTION)
                clip_tensor = increment_clip_window(clip_tensor, clip_format_frame)

            action = agent.get_action(depth_frame, clip_tensor, clip_search_enabled, obs['pov'], args.seed)
            obs, _, _, _ = env.step(action)

            depth_frame = depth_anything.depthAnything_rgb_to_depth_frame_greyscaled(obs['pov'])

            video_writer.write(cv2.cvtColor(obs['pov'], cv2.COLOR_RGB2BGR))
            prog_evaluator.update(obs)
            env.render()

    # Save Results
    video_writer.release()
    prog_evaluator.print_results()
    with open(f'{args.output_dir}/programmatic_results/programmatic_results_{str(args.run)}.txt', 'w') as f:
        for prog_task in prog_evaluator.prog_values.keys():
            f.write(f'{prog_task}: {prog_evaluator.prog_values[prog_task]}\n')
    with open(f'{args.output_dir}/agent_logs/agent_log_{str(args.run)}.txt', 'w') as f:
        for m in agent.search_log:
            f.write(m + '\n')
    with open(f'{args.output_dir}/agent_diff_logs/agent_diff_log_{str(args.run)}.txt', 'w') as f:
        for m in agent.diff_log:
            f.write(str(m) + '\n')


def increment_clip_window(window, new_frame):
    new_frame = torch.tensor(new_frame)
    new_frame = new_frame.permute(2, 1, 0) # reshape to match window

    window = window.squeeze(0)
    window = window[1:] # remove first frame from window

    window = torch.cat((window, new_frame.unsqueeze(0)), dim=0)
    return window.unsqueeze(0)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-frames', type=int, default=1*60*20)
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--search-folder', type=str, default='weights/ts_bc/')
    parser.add_argument('--title', type=str, default='recording')
    parser.add_argument('--run', type=int, default='0')

    parser.add_argument('--goal', type=str, default='gather wood')
    parser.add_argument('--distance-fn', type=str, default='euclidean', choices=DISTANCE_FUNCTIONS.keys())

    args = parser.parse_args()

    main(args)
