import cv2
import torch
import minerl
import argparse
import skvideo.io
import numpy as np
from tqdm import tqdm
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

from VPTDataset import VPTDataset
from VPTDatasetDepthAnything import VPTDatasetDepthAnything

from DepthAnythingImpl import DepthAnythingImpl

from programmatic_eval import ProgrammaticEvaluator
from distance_fns import DISTANCE_FUNCTIONS
from ModifiedTSBC_VAE_ONLY import TargetedSearchAgent

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
    depth_anything = DepthAnythingImpl()
    video_writer = cv2.VideoWriter(f'{args.output_dir}/frame_comparisons/situational_similarity/agent_recordings/{args.goal}_{args.seed}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 360))
    depth_video_writer = cv2.VideoWriter(f'{args.output_dir}/frame_comparisons/situational_similarity/agent_recordings/{args.goal}_{args.seed}_depth.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 360))

    # HumanSurvival is a sandbox `gym` environment for Minecraft with no set goal or timeframe
    env = HumanSurvival(**ENV_KWARGS).make()
    if args.seed is not None:
        env.seed(args.seed)

    agent = TargetedSearchAgent(env, search_folder=args.search_folder, distance_fn='euclidean', device=args.device)
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
            depth_video_writer.write(depth_frame) # optionally apply color if output is greyscaled
            prog_evaluator.update(obs)
            env.render()

    # Save Results of agent run
    video_writer.release()
    depth_video_writer.release()
    prog_evaluator.print_results()

    # Filter for x (20) closest points found by the model
    possible_trajectories = agent.possible_trajectories_frame_20.to('cpu').numpy()
    smallest_20_indices = np.argsort(possible_trajectories)[:10]
    largest_20_indices = np.argsort(possible_trajectories)[-20:]
    largest_20_indices = largest_20_indices[::-1] # reverse order
    goal_distances = agent.goal_distances.to('cpu').numpy()

    # rank closest trajectories and parse with starting frame, minimum of sliding window, recording_name
    search_ranking_dict = {}
    for i in range(len(smallest_20_indices)):
        traj_follow_frames = frames_until_window_min(goal_distances, smallest_20_indices[i], agent.goal_rolling_window_size)
        search_ranking_dict[i] = [smallest_20_indices[i], traj_follow_frames, None, round(float(possible_trajectories[smallest_20_indices[i]]), 2)] # rank: start_idx, end_idx, recording_name, dist
    map_found_latents_to_videos(search_ranking_dict, agent.episode_actions.episode_starts)
    calculate_episode_start(search_ranking_dict, agent.episode_actions.episode_starts)

    search_ranking_dict_largest = {}
    #largest_20_indices = agent.TEST_TOP20 ###TO REMOVE, JUST TESTING MINECLIP PATCH SEARCH
    for i in range(len(largest_20_indices)):
        traj_follow_frames = frames_until_window_min(goal_distances, largest_20_indices[i], agent.goal_rolling_window_size)
        search_ranking_dict_largest[i] = [largest_20_indices[i], traj_follow_frames, None, round(float(possible_trajectories[largest_20_indices[i]]), 2)] # rank: start_idx, end_idx, recording_name, dist
    map_found_latents_to_videos(search_ranking_dict_largest, agent.episode_actions.episode_starts)
    calculate_episode_start(search_ranking_dict_largest, agent.episode_actions.episode_starts)
    create_frame_comparison_depth(args, search_ranking_dict, search_ranking_dict_largest, "EUCLIDEAN")

    with open(f'{args.output_dir}/frame_comparisons/situational_similarity/logs/programmatic_results.txt', 'w') as f:
        for prog_task in prog_evaluator.prog_values.keys():
            f.write(f'{prog_task}: {prog_evaluator.prog_values[prog_task]}\n')
    with open(f'{args.output_dir}/frame_comparisons/situational_similarity/logs/agent_log.txt', 'w') as f:
        for m in agent.search_log:
            f.write(m + '\n')
    with open(f'{args.output_dir}/frame_comparisons/situational_similarity/logs/agent_diff_log.txt', 'w') as f:
        for m in agent.diff_log:
            f.write(str(m) + '\n')


def increment_clip_window(window, new_frame):
    new_frame = torch.tensor(new_frame)
    new_frame = new_frame.permute(2, 1, 0) # reshape to match window

    window = window.squeeze(0)
    window = window[1:] # remove first frame from window

    window = torch.cat((window, new_frame.unsqueeze(0)), dim=0)
    return window.unsqueeze(0)


def frames_until_window_min(goal_distances, latent_idx, window_size):
    return latent_idx + np.argmin(goal_distances[latent_idx : latent_idx+window_size])


# find recording of latent based on 'global' latent space
def map_found_latents_to_videos(search_rank_dict, episode_starts):
    for entry in range(len(search_rank_dict)):
        for episode in range(len(episode_starts)):
            if int(search_rank_dict[entry][0]) > int(episode_starts[episode][1]):
                search_rank_dict[entry][2] = episode_starts[episode][0]


# transforms 'global' latent idx to 'local' starting idx based on the recording
def calculate_episode_start(search_rank_dict, episode_starts):
    for entry in range(len(search_rank_dict)):
        for episode in range(len(episode_starts)):
            if search_rank_dict[entry][2] == episode_starts[episode][0]:
                follow_frames_amount = int(search_rank_dict[entry][1]) - int(search_rank_dict[entry][0])
                episode_starting_frame = int(search_rank_dict[entry][0]) - int(episode_starts[episode][1])
                episode_ending_frame = episode_starting_frame + follow_frames_amount
                search_rank_dict[entry][0] = int(episode_starting_frame)
                search_rank_dict[entry][1] = int(episode_ending_frame)
        

def create_frame_comparison_depth(args, lowest, largest, measure):

    dataset = VPTDataset()
    dataset_depthAnything = VPTDatasetDepthAnything()
    agent_recording = skvideo.io.vread(f'{args.output_dir}/frame_comparisons/situational_similarity/agent_recordings/{args.goal}_{args.seed}.mp4')
    agent_recording_depth = skvideo.io.vread(f'{args.output_dir}/frame_comparisons/situational_similarity/agent_recordings/{args.goal}_{args.seed}_depth.mp4')

    agent_obs = np.squeeze(agent_recording[19:20])
    cv2.putText(agent_obs, f"Distance Measure: {measure}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(agent_obs, f"Goal: {args.goal}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(agent_obs, f"Seed: {args.seed}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    agent_obs_depth = np.squeeze(agent_recording_depth[19:20])
    agent_horizontally = np.concatenate([agent_obs, agent_obs_depth], axis=1)
    agent_horizontally_double = np.concatenate([agent_horizontally, agent_horizontally], axis=1)
    demonstration_list = []

    print(f'### Creating frame comparison for goal: {args.goal}')
    print(f"LOWEST {measure} DICT")
    print(lowest)
    print()
    print(f"LARGEST {measure} DICT")
    print(largest)
    for rank in range(5):

        vid, _, _ = dataset.get_from_vid_id(largest[rank][2])
        vid_lowest, _, _ = dataset.get_from_vid_id(lowest[rank][2])

        vid_depth, _, _ = dataset_depthAnything.get_from_vid_id('dataset/' + largest[rank][2])
        vid_depth_lowest, _, _ = dataset_depthAnything.get_from_vid_id('dataset/' + lowest[rank][2])

        demonstration = np.squeeze(vid[largest[rank][0]:largest[rank][0] + 1])
        demonstration_depth = np.squeeze(vid_depth[largest[rank][0]:largest[rank][0] + 1])
        demonstration_lowest = np.squeeze(vid_lowest[lowest[rank][0]:lowest[rank][0] + 1])
        demonstration_depth_lowest = np.squeeze(vid_depth_lowest[lowest[rank][0]:lowest[rank][0] + 1])

        # Adding text annotations
        cv2.putText(demonstration, f"Rank: {rank}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(demonstration, f"Goal: {args.goal}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(demonstration, f"Seed: {args.seed}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(demonstration, f"Distance: {largest[rank][3]}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(demonstration, f"Video: {largest[rank][2].rsplit('/', 1)[-1]}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(demonstration, f"Starting Frame: {largest[rank][0]}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(demonstration_lowest, f"Rank: {rank}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(demonstration_lowest, f"Goal: {args.goal}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(demonstration_lowest, f"Seed: {args.seed}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(demonstration_lowest, f"Distance: {lowest[rank][3]}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(demonstration_lowest, f"Video: {lowest[rank][2].rsplit('/', 1)[-1]}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(demonstration_lowest, f"Starting Frame: {lowest[rank][0]}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        demonstration_horizontally = np.hstack((demonstration, demonstration_depth))
        demonstration_horizontally_lowest = np.hstack((demonstration_lowest, demonstration_depth_lowest))
        demonstration_horizontally_total = np.hstack((demonstration_horizontally, demonstration_horizontally_lowest))
        demonstration_list.append(demonstration_horizontally_total)

    # Save the frame as a PNG image
    vertical = np.vstack(demonstration_list)
    horizontal = np.vstack((agent_horizontally_double, vertical))
    cv2.imwrite(f'output/frame_comparisons/situational_similarity/{args.seed}_{args.goal}.png', cv2.cvtColor(horizontal, cv2.COLOR_RGB2BGR))



# creates recording with observation used for search
# displays used params at top
# bottom shows trajectorie from closest latent found in search until min. distance within sliding window
def create_recording(args, search_ranking_dict):

    dataset = VPTDataset()
    agent_recording = skvideo.io.vread(f'{args.output_dir}/search_analysis/agent_recordings/{args.goal}_{args.seed}.mp4')

    print(f'### Creating Video for goal: {args.goal}')
    for rank in tqdm(range(len(search_ranking_dict))):
        duration_until_goal = search_ranking_dict[rank][1] - search_ranking_dict[rank][0]
        vid, _, _ = dataset.get_from_vid_id(search_ranking_dict[rank][2])
        dataset_video = np.empty((duration_until_goal, 360, 640, 3), dtype=np.uint8)
        dataset_video[0:duration_until_goal] = vid[search_ranking_dict[rank][0]:search_ranking_dict[rank][1]]

        agent_observation = np.empty((duration_until_goal, 360, 640, 3), dtype=np.uint8)
        for i in range(duration_until_goal):
            agent_observation[i] = agent_recording[19:20]

        video_writer = cv2.VideoWriter(f'{args.output_dir}/search_analysis/rankings/GOAL_{args.goal}/SEED_{args.seed}/{rank}_{args.goal}_{args.seed}_{search_ranking_dict[rank][3]}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 2*360))
        comb_video = np.concatenate([agent_observation, dataset_video], axis=1)

        for frame in comb_video:
            cv2.rectangle(dataset_video[i], (0, 0), (300, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"Rank: {rank}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Goal: {args.goal}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Seed: {args.seed}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Distance: {search_ranking_dict[rank][3]}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Video: {search_ranking_dict[rank][2]}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Starting Frame: {search_ranking_dict[rank][0]}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video_writer.release()


def create_frame_comparison(args, search_ranking_dict):

    dataset = VPTDataset()
    agent_recording = skvideo.io.vread(f'{args.output_dir}/search_analysis/agent_recordings/{args.goal}_{args.seed}.mp4')

    print(f'### Creating frame comparison for goal: {args.goal}')
    for rank in tqdm(range(len(search_ranking_dict))):
        duration_until_goal = 1
        vid, _, _ = dataset.get_from_vid_id(search_ranking_dict[rank][2])
        dataset_video = np.empty((duration_until_goal, 360, 640, 3), dtype=np.uint8)
        dataset_video[0:duration_until_goal] = vid[search_ranking_dict[rank][0]:search_ranking_dict[rank][0] + 1]

        agent_observation = np.empty((duration_until_goal, 360, 640, 3), dtype=np.uint8)
        for i in range(duration_until_goal):
            agent_observation[i] = agent_recording[19:20]

        # Combine agent observation and dataset video frames vertically
        comb_video = np.concatenate([agent_observation, dataset_video], axis=1)

        # Since duration_until_goal is 1, we only have one frame to work with
        frame = comb_video[0]

        # Add the text annotations
        cv2.putText(frame, f"Rank: {rank}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Goal: {args.goal}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Seed: {args.seed}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Distance: {search_ranking_dict[rank][3]}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Video: {search_ranking_dict[rank][2]}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Starting Frame: {search_ranking_dict[rank][0]}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Save the frame as a PNG image
        output_path = f'{args.output_dir}/search_analysis/rankings/GOAL_{args.goal}/SEED_{args.seed}/{rank}_{args.goal}_{args.seed}_{search_ranking_dict[rank][3]}.png'
        cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)

