import os
import sys
import json
import requests
import skvideo.io
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

import traceback

VIDEO_TYPE = '.mp4'
LABEL_TYPE = '.jsonl'

class VPTDatasetDepthAnything(Dataset):
    def __init__(self, index_file='dataset/chop_tree_handpicked_01-20_depth.json', base_dir='dataset/data/depth_videos/', download_all=False):
        """Dataset class for the videos and actions used for preprocessed OpenAI's VPT with DepthAnything.

        Args:
        index_file: File path to the index file (from VPT's GitHub).
        base_dir: Folder path to the base directory where videos and actions are saved.
        download_all: If videos and actions should be downloaded when instantiating (else greedy when accessing data[idx]).
        """

        with open(index_file, 'r') as f:
            data = json.load(f)
        self.download_base_url = data['basedir']
        self.data = np.array(list(map(lambda p: p.rsplit('.', 1)[0], data['relpaths'])))

        self.base_dir = base_dir
        self.data_dir = self.base_dir + self.data[0].rsplit('/', 1)[0]

        os.makedirs(os.path.dirname(self.base_dir + self.data[0]), exist_ok=True)

        if download_all:
            print('Downloading Dataset...')
            for idx in tqdm(range(len(self.data))):
                self.download(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.base_dir + self.data[idx]
        print('dataset get file path', file_path)

        try:
            # Download if it does not exist
            if not os.path.isfile(file_path + VIDEO_TYPE) or not os.path.isfile(file_path + LABEL_TYPE):
                print('will download')
                self.download(idx)

            # Load videos and actions
            print('will read')
            video = skvideo.io.vread(file_path + VIDEO_TYPE)
            print('will open json')
            with open(file_path + LABEL_TYPE, 'r', encoding='utf-8') as f:
                actions = [json.loads(line) for line in f]
        except Exception as e:
            print('except')
            print(e)
            traceback.print_exc() 
            
            print('idx:', idx)
            cur_line = None
            try:
                with open(file_path + LABEL_TYPE, 'r', encoding='utf-8') as f:
                    for line in f:
                        cur_line = json.loads(line)
            except:
                print('cur_line:', cur_line)
            
            return None, None, None

        if len(video) <= 100:
            print('vid<=100')
            return None, None, None

        print('return', self.data[idx])
        return video, actions, self.data[idx]

    def get_from_vid_id(self, vid_id):
        print(vid_id.split('/')[-1])
        for idx in range(len(self)):
            if self.data[idx] == vid_id.split('/')[-1]:
                return self[idx]
        return None, None, None


    def delete(self, vid_id):
        vid_name = vid_id.rsplit('/', 1)[-1]
        for f in os.listdir(self.data_dir):
            if f.endswith(vid_name + VIDEO_TYPE) or f.endswith(vid_name + LABEL_TYPE):
                os.remove(os.path.join(self.data_dir, f))
    
    def download(self, idx):
        url = self.download_base_url + self.data[idx]
        file_path = self.base_dir + self.data[idx]

        # Download Video if it not exists
        if not os.path.isfile(file_path + VIDEO_TYPE):
            response = requests.get(url + VIDEO_TYPE)  # TODO: What if file is not available?
            with open(file_path + VIDEO_TYPE, 'wb') as f:
                f.write(response.content)

        # Download Label if it not exists
        if not os.path.isfile(file_path + LABEL_TYPE):
            response = requests.get(url + LABEL_TYPE)
            with open(file_path + LABEL_TYPE, 'wb') as f:
                f.write(response.content)
