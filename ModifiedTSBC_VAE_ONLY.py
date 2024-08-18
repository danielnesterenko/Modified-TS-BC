import cv2
import torch
import minerl
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import torchvision.utils as vutils
import sys
import os

from VPTDataset import VPTDataset
from CVAE import load_cvae
from EpisodeActions import EpisodeActions
from LatentSpaceMineCLIP_Patches import LatentSpaceMineCLIP, load_mineclip, SLIDING_WINDOW_SIZE
from LatentSpaceDepthVAE import LatentSpaceDepthVAE, AGENT_RESOLUTION
from evaluation_utils import frames_until_window_min, map_found_latents_to_videos, calculate_episode_start
import util_functions

import VAE.build_latent_space_vae as VAE

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((216, 384))  # Resize to the desired dimensions
])

class TargetedSearchAgent():
    def __init__(self, env, search_folder='weights/ts_bc/', max_follow_frames=20, goal_rolling_window_size=60*20, distance_fn='euclidean', device='cuda'):
        self.env = env
        self.past_frames = []
        self.frame_counter = 0  # How many frames the agent has played
        self.follow_frame = -1  # How many frames from the trajectory we have followed
        self.max_follow_frames = max_follow_frames  # How many frames we can follow before searching for a new trajectory
        self.max_clip_follow_frames = 100
        self.trigger_execution_phase = False
        self.clip_follow_counter = -1
        self.count_clip_searches = 0 # Only needed for frame_comparison
        self.clip_sim_threshold = 0.7


        self.redo_search_counter = 0  # TODO better name?
        self.redo_search_threshold = 3
        self.diff_threshold = 300000 # Adjust threshold here

        self.diff_log = []
        self.search_log = []
        self.device = device

        self.episode_actions = EpisodeActions().load(search_folder=search_folder, N=20)
        self.latent_space_mineclip = LatentSpaceMineCLIP(distance_fn=distance_fn, device=self.device).load(self.episode_actions, latents_folder=search_folder+'latents_mineclip_patches/')
        self.latent_space_vae = LatentSpaceDepthVAE(distance_fn=distance_fn, device=self.device).load(self.episode_actions, latents_folder=search_folder+'latents_4x4x8_128/')
        self.latents = self.latent_space_vae.latents

        self.mineclip_model = load_mineclip(device=self.device)
        self.cvae_model = load_cvae(device=self.device)

        self.same_episode_penalty = torch.zeros(len(self.episode_actions.actions)).to(self.device)
        self.select_same_penalty = 10.0  # TODO

        self.nearest_idx = None
        self.current_goal = None
        self.future_goal_distances = None
        self.future_goal_distances_normalized = None
        self.trajectory_filter_top25p = None
        self.trajectory_filter_top05p = None
        self.goal_rolling_window_size = goal_rolling_window_size
        
        self.possible_trajectories_frame_20 = None
        self.latent_space_vpt_frame_20 = None
        self.mineclip_latents = self.latent_space_mineclip.latents
        self.goal_distances = None # distance between 'text-prompt-latent' and mineCLIP latents

        ##TESTING - TO REMOVE
        self.TEST_OBS = None
        self.TEST_TOP20 = None
    
    def set_goal(self, text_goal):
        '''
        Compute the `future_goal_distances` array that show
        how far near the goal an agent can become when following 
        a certain dataset episode frame for some time.
        '''
        self.current_goal = text_goal
        text_latent = self.mineclip_model.encode_text(self.current_goal) # encode goal text prompt
        vis_text_latent = self.cvae_model(text_latent) # transform encoded text goal to a 'visual encoding' that can be represented in mineCLIP latent space
        self.goal_distances = self.latent_space_mineclip.get_distances(vis_text_latent[0].detach()) # get distances between all latents of mineCLIP's indexed space and the 'visual encoding' based on the goal text prompt
        self.latent_space_mineclip.min_distances, self.latent_space_mineclip.min_indices = torch.topk(self.goal_distances, 50, largest=False)
        self.latent_space_mineclip.get_goal_patch_mask()
        
        goal_distances_padding = F.pad(self.goal_distances, (0, self.goal_rolling_window_size-1), 'constant', float('inf'))
        goal_distances_rolling = goal_distances_padding.unfold(0, self.goal_rolling_window_size, 1)
        self.future_goal_distances = goal_distances_rolling.min(1).values
        self.future_goal_distances_normalized = util_functions.min_max_normalize(self.future_goal_distances)
        self.future_goal_distances_normalized = util_functions.scale_to_range(self.future_goal_distances_normalized, 0, 10)

        

        mn, mean = self.future_goal_distances.min(), self.future_goal_distances.mean()
        top25p = mean-(mean-mn)/2
        top05p = mean-(mean-mn)/10*9
        self.trajectory_filter_top25p = self.future_goal_distances >= top25p
        self.trajectory_filter_top05p = self.future_goal_distances >= top05p

        print(f'Set new goal: \"{self.current_goal}\" with {len(self.trajectory_filter_top05p)-self.trajectory_filter_top05p.sum()}/{len(self.trajectory_filter_top05p)} latents')

    def get_action(self, obs, clip_tensor, clip_search_enabled, rgb_obs, seed):
        self.TEST_OBS = obs
        clip_obs_mask_dist = None
        self.follow_frame += 1
        self.frame_counter += 1

        latent = self.get_latent(obs)

        #===============Only executed in first 20 frames===============
        if self.frame_counter < 20:  # Warmup phase, turning around
            action = self.env.action_space.noop()
            action['camera'] = [0, 20]
            self.diff_log.append(0)  # TODO remove
            return action

        if self.frame_counter == 20:
            self.search(latent, True)
        #==============================================================

        # enable execution phase if current observation meets MineCLIP patch similarity threshold and not currently in execution phase
        if clip_search_enabled and self.clip_follow_counter == -1:
            observation_masked = self.get_masked_observation(clip_tensor)
            self.trigger_execution_phase, execution_traj_indices = self.evaluate_masked_similarity(observation_masked, rgb_obs, seed) # Bool | Array containing traj. indeces with highest mask similarity
            self.nearest_idx = execution_traj_indices[0]
            if self.trigger_execution_phase:
                self.log_new_nearest()

        if self.trigger_execution_phase:
            self.clip_follow_counter += 1
            action, is_null_action = self.episode_actions.actions[self.nearest_idx + self.clip_follow_counter]

            if self.clip_follow_counter == self.max_clip_follow_frames:
                self.clip_follow_counter = -1
            
            return action

        if self.should_search_again(latent):
            self.search(latent)

        action, is_null_action = self.episode_actions.actions[self.nearest_idx + self.follow_frame]

        return action
    

    def evaluate_masked_similarity(self, obs_masked, rgb_obs, seed):
        '''
        obs_masked -> torch.size([num_mask_patches, patch_embeddings])
        goal_traj_masked -> torch.size([num_goal_images, num_mask_patches, patch_embeddings])
        '''
        agreed_patches_per_traj = []
        mask_patch_indeces = self.latent_space_mineclip.task_patch_indices #contains indeces of patches [0, 161] which where chosen for mask
        goal_traj_masked = self.latent_space_mineclip.goal_trajectories_patches_masked
        num_images, num_patches, num_embeddings = goal_traj_masked.shape
        similarity_count = np.zeros(num_images)

        patch_sim_threshold = 0.7
        for image_i in range(num_images):
            agreed_patches = []
            for patch_j in range(num_patches):
                similarity_ij = F.cosine_similarity(obs_masked[patch_j].unsqueeze(0), goal_traj_masked[image_i][patch_j])
                if similarity_ij > patch_sim_threshold:
                    agreed_patches.append(mask_patch_indeces[patch_j]) # adding patch idx that exceeded sim threshhold 
                    similarity_count[image_i] += 1 # image sim counter +1 for each patch that exceeds threshhold of similarity between observation and demonstration
            agreed_patches_per_traj.append(agreed_patches)

        patch_agreement_count = 13
        if np.max(similarity_count) >= patch_agreement_count:
            #print(f'### Matching patches: {np.max(similarity_count)}')
            print('### Execution mode triggered.')
            agreed_patches = agreed_patches_per_traj[int(np.argmax(similarity_count))] # patches that exceeded threshold of all patches from mask
            top_x_indices = np.argsort(similarity_count)[-3:][::-1]
            #print('TOP X')
            #print(f'Goal-trajectory indices: {top_x_indices}') # indeces of top x trajectories from prefiltered list containing indices of top traj. from all traj.
            self.create_frame_comparison(top_x_indices, rgb_obs, agreed_patches, seed) # creates frame comparison on moment when execution mode was triggered.

            return True, self.latent_space_mineclip.min_indices[top_x_indices]
        else:
            return False, self.latent_space_mineclip.min_indices
        

    def create_frame_comparison(self, goal_indeces, rgb_obs, agreed_patches, seed):
        dataset = VPTDataset()
        search_ranking_dict = {}

        for i in range(len(goal_indeces)):
            goal_distances = self.goal_distances.cpu().numpy()
            traj_follow_frames = frames_until_window_min(goal_distances, goal_indeces[i], self.goal_rolling_window_size)
            search_ranking_dict[i] = [self.latent_space_mineclip.min_indices[goal_indeces[i]], traj_follow_frames, None, 1] ##
        map_found_latents_to_videos(search_ranking_dict, self.episode_actions.episode_starts)
        calculate_episode_start(search_ranking_dict, self.episode_actions.episode_starts)
        print(search_ranking_dict)
        
        frames_hstacked = []
        for frame in range(len(search_ranking_dict)):
            vid, _, name = dataset.get_from_vid_id(search_ranking_dict[frame][2])
            demonstration = vid[search_ranking_dict[frame][0]]
            
            '''
            can be utilized to draw patches over frame comparison, however its unsure if the positions are correct.

            if frame == 0:
                obs_patch_img, dem_patch_img = self.draw_patches_on_img(rgb_obs, demonstration, agreed_patches)
                patches_hstacked = np.hstack((obs_patch_img, dem_patch_img))
                cv2.imwrite(f'./output/frame_comparisons/execution_mode/executionModeTrigger_patches_{seed}_{self.count_clip_searches}.png', cv2.cvtColor(patches_hstacked, cv2.COLOR_RGB2BGR))
            '''

            frames_hstacked.append(np.hstack((rgb_obs, demonstration)))
            

        frames_vstacked = np.vstack(frames_hstacked) 
        cv2.imwrite(f'./output/frame_comparisons/execution_mode/executionModeTrigger_{seed}_{self.count_clip_searches}.png', cv2.cvtColor(frames_vstacked, cv2.COLOR_RGB2BGR))
        self.count_clip_searches += 1


    def draw_patches_on_img(self, rgb_obs, demonstration, agreed_patches):
        cell_width = 16
        cell_height = 16

        obs_resized = cv2.resize(rgb_obs, (256, 160))
        dem_resized = cv2.resize(demonstration, (256, 160))
        height, width, _ = obs_resized.shape

        # Draw vertical + horizontal grid lines
        for x in range(0, width, cell_width):
            cv2.line(obs_resized, (x, 0), (x, height), (0, 0, 0), 1)
        for y in range(0, height, cell_height):
            cv2.line(obs_resized, (0, y), (width, y), (0, 0, 0), 1)

        for x in range(0, width, cell_width):
            cv2.line(dem_resized, (x, 0), (x, height), (0, 0, 0), 1)
        for y in range(0, height, cell_height):
            cv2.line(dem_resized, (0, y), (width, y), (0, 0, 0), 1)

        # List of cell indices to color edges in green
        highlight_cells_agree = agreed_patches
        highlight_cells_disagree = [item for item in self.latent_space_mineclip.task_patch_indices if item not in agreed_patches]


        # Number of cells per row
        cells_per_row = width // cell_width

        # Draw Agreements (Green)
        for cell_index in highlight_cells_agree:
            row = cell_index // cells_per_row
            col = cell_index % cells_per_row
            top_left_x = col * cell_width
            top_left_y = row * cell_height
            bottom_right_x = top_left_x + cell_width
            bottom_right_y = top_left_y + cell_height

            cv2.rectangle(obs_resized, 
                        (top_left_x, top_left_y), 
                        (bottom_right_x, bottom_right_y), 
                        (0, 255, 0), 1)
            cv2.rectangle(dem_resized, 
                        (top_left_x, top_left_y), 
                        (bottom_right_x, bottom_right_y), 
                        (0, 255, 0), 1)

        # Draw Disagreements (Red)
        for cell_index in highlight_cells_disagree:
            row = cell_index // cells_per_row
            col = cell_index % cells_per_row
            top_left_x = col * cell_width
            top_left_y = row * cell_height
            bottom_right_x = top_left_x + cell_width
            bottom_right_y = top_left_y + cell_height

            cv2.rectangle(obs_resized, 
                        (top_left_x, top_left_y), 
                        (bottom_right_x, bottom_right_y), 
                        (255, 0, 0), 1)
            cv2.rectangle(dem_resized, 
                        (top_left_x, top_left_y), 
                        (bottom_right_x, bottom_right_y), 
                        (255, 0, 0), 1)

        return obs_resized, dem_resized


    def get_masked_observation(self, clip_tensor):
        clip_tensor = clip_tensor.to(self.device)
        latent, patch_embeddings = self.mineclip_model.encode_video(clip_tensor)
        patch_embeddings = patch_embeddings.cpu().numpy()
        patch_embeddings_numpy = np.array(patch_embeddings[15][self.latent_space_mineclip.task_patch_indices])
        patch_embeddings_masked = torch.tensor(patch_embeddings_numpy).to(self.device)
        return patch_embeddings_masked
    
    
    def get_latent(self, frame):
        #frame = self.preprocess(frame, 256)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_preprocessed = VAE.preprocess_img(frame)

        latent = VAE.encode_frame(frame_preprocessed)
        latent = latent.squeeze(0)
        latent = latent.detach().numpy()
        
        del(frame)
        return torch.tensor(latent).cuda()
    
    def get_latent_mineclip(self, obs):
        frame = obs['pov']
        frame = np.transpose(cv2.resize(frame, AGENT_RESOLUTION), (2, 1, 0))
        self.past_frames.append(frame)
        if len(self.past_frames) > SLIDING_WINDOW_SIZE:
            self.past_frames.pop(0)

        frame_window = torch.from_numpy(np.array(self.past_frames)).unsqueeze(0).to(self.device)
        latent = self.mineclip_model.encode_video(frame_window)[0]
        del(frame_window)

        return latent
    
    def search(self, latent, flag=False):
        if self.current_goal is None:
            raise Exception('Goal is not set.')
    
        # Reduce the penalty for choosing the same episode
        self.same_episode_penalty = torch.maximum(self.same_episode_penalty - 1, torch.tensor(0))

        # Search for the next trajectory based on the goal and current state
        possible_trajectories = self.latent_space_vae.get_distances(latent)

        # Saving possible_trajectories at frame 20 to analyze latent space manually
        if flag:
            self.possible_trajectories_frame_20 = possible_trajectories
            self.latent_space_vpt_frame_20 = self.latent_space_vae.get_distances(latent)

        #possible_trajectories = self.trajectory_filter_top05p*300 + self.latent_space_vpt.get_distances(latent)
        possible_trajectories += self.same_episode_penalty
        self.nearest_idx = possible_trajectories.argmin().to('cpu').item()

        # Give a penalty to the episode (TODO currently window around) that has been chosen
        self.same_episode_penalty[max(self.nearest_idx-5, 0):self.nearest_idx+5] = self.select_same_penalty

        self.follow_frame = -1
        self.redo_search_counter = 0

        self.log_new_nearest()
    
    def should_search_again(self, latent):
        self.calc_follow_difference(latent)
        
        # Trigger a new search if
        #   1. there has not been a search yet,
        #   2. we followed a trajectory for too long,
        #   3. the reference trajectory ends, or
        #   4. the divergence between our state and the reference is too high.
        return self.nearest_idx is None \
            or self.follow_frame > self.max_follow_frames \
            or self.episode_actions.is_last(self.nearest_idx + self.follow_frame) \
            or self.redo_search_counter >= self.redo_search_threshold
        
    def calc_follow_difference(self, latent):
        if self.nearest_idx is None:
            self.diff_log.append(0)
            return

        # Compute current difference
        diff_to_follow_latent = self.latent_space_vae.get_distance(self.nearest_idx + self.follow_frame, latent)
        self.diff_log.append(diff_to_follow_latent.to('cpu').item())

        # After selecting a new trajectory, follow it for the first few frames
        if self.follow_frame < 0.33 * self.max_follow_frames:
            return  # TODO put back in prior `if`

        # Increase or decrease the counter based on the difference to the reference
        if diff_to_follow_latent > self.diff_threshold:
            self.redo_search_counter += 1
        else:
            self.redo_search_counter = max(self.redo_search_counter - 1, 0)

    def log_new_nearest(self):
        episode = 0
        while episode+1 < len(self.episode_actions.episode_starts) and int(self.episode_actions.episode_starts[episode+1][1]) <= self.nearest_idx:  # TODO remove int()
            episode += 1
        episode_id, episode_start = self.episode_actions.episode_starts[episode]
        episode_start = int(episode_start)

        episode_frame = self.nearest_idx - episode_start + SLIDING_WINDOW_SIZE - 1
        self.search_log.append(f'[Frame {self.frame_counter:04} ({self.frame_counter // 20 // 60}:{(self.frame_counter // 20) % 60:02})] Found nearest in {episode_id} at frame {episode_frame} ({episode_frame // 20 // 60}:{(episode_frame // 20) % 60:02})')

    def preprocess(self, frame, batch_size):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
        frame_tensor = transform(frame)
        frames_ret = []

        frames_ret.append(frame_tensor)
        video_tensor = torch.stack(frames_ret)
        return video_tensor.to(self.device)
