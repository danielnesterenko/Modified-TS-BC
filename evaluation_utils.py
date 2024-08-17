import numpy as np

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


def frames_until_window_min(goal_distances, latent_idx, window_size):
    return latent_idx + np.argmin(goal_distances[latent_idx : latent_idx+window_size])