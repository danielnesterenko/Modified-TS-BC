import cv2
import torch
import sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
#from mineclip import MineCLIP
from MineCLIP.mineclip.mineclip import MineCLIP
import torch.nn.functional as F

from distance_fns import DISTANCE_FUNCTIONS

AGENT_RESOLUTION = (256, 160)  # (W, H)
SLIDING_WINDOW_SIZE = 16

class LatentSpaceMineCLIP:
    def __init__(self, distance_fn='euclidean', device='cuda'):
        self.latents = []  # Python List while training, Numpy array while inference
        self.patch_embeddings_unstacked = []
        self.min_distances = None
        self.min_indices = None
        self.task_patch_indices = None
        self.goal_trajectories_patches_masked = None
        self.num_goal_patches_mask = 20
        self.distance_function = DISTANCE_FUNCTIONS[distance_fn]
        self.device = device
    
    
    def get_goal_patch_mask(self):
        '''
        Filters MineCLIP latent space based on goal condition and
        finds patch-indeces that are most similar most often
        to build a mask.
        '''
        
        # Bring Patches into correct shape
        self.min_distances = self.min_distances.to('cpu').numpy()
        self.min_indices = self.min_indices.to('cpu').numpy().astype('uint32')
        self.min_indices = np.array(self.min_indices, dtype=int)
        self.patch_embeddings_unstacked = np.array(self.patch_embeddings_unstacked)

        # Pre-Filter all patch embeddings by goal trajectories to lower memory for computation
        self.patch_embeddings_unstacked = [item for sublist in self.patch_embeddings_unstacked for item in sublist]
        self.patch_embeddings_unstacked = np.array(self.patch_embeddings_unstacked)
        goal_trajectories_patches = self.patch_embeddings_unstacked[self.min_indices] # lenghts: goal_trajectories -> 50, goal_trajectories[0] -> 161, goal_trajectories[0][0] -> 768
        goal_trajectories_patches = torch.from_numpy(goal_trajectories_patches).to(self.device)

        # Finding relevant goal patches for masking
        self.task_patch_indices = self.find_task_patches(goal_trajectories_patches)
        self.goal_trajectories_patches_masked = goal_trajectories_patches[:, self.task_patch_indices, :].to(self.device)


    def find_task_patches(self, embeddings):
        patch_similarity_count = np.zeros(161)
        num_images, num_patches, num_embeddings = embeddings.shape

        patch_sim_threshold = 0.9
        print('### Searching for goal related mask ...')
        for patch_i in range(num_patches):
            for image_j in range(num_images):
                for image_jk in range(num_images):
                    # compare 1-patch along all images, 2-patch...
                    # if t > 0.9 -> inc. counter
                    # pick patches with highest counter
                    similarity_ij = F.cosine_similarity(embeddings[image_j][patch_i].unsqueeze(0), embeddings[image_jk][patch_i].unsqueeze(0))
                    if similarity_ij >= patch_sim_threshold:
                        patch_similarity_count[patch_i] += 1

        #print(patch_similarity_count)
        goal_related_patches = np.argsort(patch_similarity_count)[-self.num_goal_patches_mask:][::-1]
        print(f'### Found goal related patches for mask: {goal_related_patches.tolist()}')
        return goal_related_patches.tolist()

    
    def find_task_patches_OLD(self, embeddings):
        
        #Finds task related patches for a filtered amount of 
        #patch embeddings.
        
        _, num_patches, _ = embeddings.shape
        patch_similarity_count = torch.zeros(num_patches)
        
        for patch_idx in range(num_patches):
            patch_embeddings = embeddings[:, patch_idx, :]  # Extract patch embeddings for current patch across all the images
            
            # Compute pairwise cosine similarities
            similarities = torch.nn.functional.cosine_similarity(
                patch_embeddings.unsqueeze(1),
                patch_embeddings.unsqueeze(0),
                dim=2
            )
            
            similarities.fill_diagonal_(-float('inf')) # We don't want to compare a patch to itself

            most_similar_patch_idx = similarities.argmax(dim=1)
            
            # Count how often a patch was most similar
            for idx in most_similar_patch_idx:
                patch_similarity_count[idx] += 1
        
        # Filter topk most similar patches
        topk_patch_indices = patch_similarity_count.topk(self.num_goal_patches_mask).indices #adjust num of patches here
        print(topk_patch_indices)
        sys.exit()
        return topk_patch_indices.tolist()
    


    @torch.no_grad()
    def load(self, episode_actions, latents_folder='weights/ts_bc/latents_mineclip_patches/'):
        for vid_id, _ in episode_actions.episode_starts:
            _, name = vid_id.rsplit('/', 1)
            vid_latents = np.load(latents_folder + name + '.npy', allow_pickle=True)
            latents = [latent[0] for latent in vid_latents]
            patch_embeddings = [latent[1] for latent in vid_latents]
            self.latents.append(latents)
            self.patch_embeddings_unstacked.append(patch_embeddings)

        self.latents = torch.from_numpy(np.vstack(self.latents)).to(self.device)
        #self.patch_embeddings = torch.from_numpy(np.vstack(self.patch_embeddings)).to(self.device) -> unfiltered is to much data, hence running into OutOfMemoryError
        print(f'Loaded MineCLIP latent space with {len(self.latents)} latents')
        return self
    
    @torch.no_grad()
    def load_OLD(self, latents_file='weights/ts_bc/latents_mineclip.npy'):  # TODO update to new format
        self.latents = torch.from_numpy(np.load(latents_file, allow_pickle=True)).to(self.device)
        print(f'Loaded MineCLIP latent space with {len(self.latents)} latents')
        return self
    
    def save(self, latents_file='weights/ts_bc/latents_mineclip'):  # TODO remove?
        latents = np.array(self.latents)
        np.save(latents_file, latents)

    @torch.no_grad()
    def train_episode(self, mineclip_model, frames, vid_id):
        episode_latents = []

        resized_frames = np.empty((frames.shape[0], AGENT_RESOLUTION[1], AGENT_RESOLUTION[0], 3), dtype=np.uint8)
        for ts in range(frames.shape[0]):
            resized_frame = cv2.resize(frames[ts], AGENT_RESOLUTION)
            resized_frames[ts] = resized_frame

        sliding_window_frames = sliding_window_view(resized_frames, SLIDING_WINDOW_SIZE, 0)
        sliding_window_frames = torch.tensor(np.transpose(sliding_window_frames, (0, 4, 3, 2, 1)))

        inter_batch_size = 1
        for i in range(sliding_window_frames.shape[0] // inter_batch_size + 1):
            inter_batch_frames = sliding_window_frames[i*inter_batch_size:(i+1)*inter_batch_size].to(self.device)

            latents = mineclip_model.encode_video(inter_batch_frames)

            latents, latent_patch_embeddings = latents
            latents = latents.to('cpu').numpy().astype('float16')
            latent_patch_embeddings = latent_patch_embeddings.to('cpu').numpy().astype('float16')

            emb_idx = list(range(latent_patch_embeddings.shape[0]))
            emb_idx_16 = emb_idx[15::16]

            for j in range(latents.shape[0]):
                latent_to_save = latents[j]
                patch_embedding_to_save = latent_patch_embeddings[emb_idx_16[j]]
                episode_latents.append((latent_to_save, patch_embedding_to_save))

            if i % 100 == 0:
                print(f'{i} batches encoded')
            
            del(inter_batch_frames)

        print(f'### Encoded {len(episode_latents)} embeddings.')
        print(f'### MineCLIP + patch embedding finished for video: {vid_id}.')
        
        np.save('./weights/ts_bc/latents_mineclip_patches/' + vid_id.rsplit('/', 1)[-1], np.array(episode_latents))

    def get_distances(self, latent):
        return self.distance_function(self.latents, latent)
    
    def get_distance(self, idx, latent):
        return self.distance_function(self.latents[idx], latent)

    def get_nearest(self, latent): # TODO episode_start is removed
        diffs = self.get_distances(latent)
        nearest_idx = diffs.argmin()#.to('cpu').item() # TODO remove .to('cpu').item()
        return nearest_idx

def load_mineclip(weights_file='weights/mineclip/attn.pth', device='cuda'):  # TODO: in it's own file?
    mineclip = MineCLIP(arch='vit_base_p16_fz.v2.t2', hidden_dim=512, image_feature_dim=512, mlp_adapter_spec='v0-2.t0', pool_type='attn.d2.nh8.glusw', resolution=[160, 256])
    mineclip.load_ckpt(weights_file, strict=True)
    mineclip.eval()

    return mineclip.to(device)
