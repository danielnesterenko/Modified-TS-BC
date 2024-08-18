# Modified TS-BC
Modified TS-BC is a modification of the [Targeted Search-Based Behavioral Cloning](https://pages.github.com/]https://github.com/JulianBvW/TS-BC) model for playing Minecraft.
It looks for the most similar situation in relation to a prompted goal between the agent and a demonstration dataset, by comparing embeddings of different embedding spaces.

See [thesis_presentation_modified_tsbc.pptx](https://github.com/user-attachments/files/16648347/thesis_presentation_modified_tsbc.pptx) for additional information.


## Disclaimer

It utilizes parts of the following projects:
* [TS-BC](https://pages.github.com/]https://github.com/JulianBvW/TS-BC)
* [MineCLIP](https://github.com/MineDojo/MineDojo)
* [MineRL](https://github.com/minerllabs/minerl)
* [STEVE-1](https://github.com/Shalev-Lifshitz/STEVE-1)
* [DepthAnything](https://github.com/LiheYoung/Depth-Anything)
* [DINOv2](https://github.com/facebookresearch/dinov2)

and is for academic reasons only.

## Installation

1. Use Java JDK 8.
2. Make a new conda environment `conda create --name m-tsbc python=3.9`
3. Install the following dependencys and projects:
* `pip install pip==21 setuptools==65.5.0 importlib-resources==1.3`
* `pip install torch torchvision torchaudio opencv-python tqdm numpy==1.23.5 pandas scikit-video`
* `pip install gym==0.21.0 gym3 attrs`
* `pip install git+https://github.com/minerllabs/minerl`
* `pip install git+https://github.com/MineDojo/MineCLIP`
4. Download the following weights and data and put inside the correct directories:
* Download [attn.pth](https://drive.google.com/file/d/1uaZM1ZLBz2dZWcn85rZmjP7LV6Sg5PZW/view) and move into weights/mineclip
* Download [steve1_prior.pt](https://drive.google.com/uc?id=1OdX5wiybK8jALVfP5_dEo0CWm9BQbDES) and move into weights/cvae
* Download [model_vid10000_4x4x8.pth](https://drive.google.com/uc?export=download&id=1588qzaRGNvQWsibVBDdp_9zA8VFHd4wE) and move into weights/pretrained_vae
* Download the directory [data](https://drive.google.com/drive/folders/1WdzwLzmUilVKSt6sYVvSZQLoy2FPHhk0?usp=sharing) and move it into dataset/
* Download the directory [facebookresearch_dinov2_main](https://drive.google.com/drive/folders/1lp5mSZWSDg0Ca7ebJmujK5YdH-NDSgZc?usp=sharing) and move it into torchhub/

## Indexing of the Embedding Space

* Run train.py with the new conda environment enabled (this process may take 2-3 hours based on your device):
* `conda activate m-tsbc`
* `python train.py`

## Run the Agent
* Model has only been ran on headless machines. The following instructions therefore are for headless machines only. Removing the prefix `xvfb-run -a` might enable the model to run on a machine with head.
* Run the agent on goals & seeds defined within `run_batch_analysis.py` by `xvfb-run -a python run_batch_analysis.py`. After finishing, its video footage will be placed inside output/agent_recordings.
* To make frame comparisons based on the VAE-embedding space run `xvfb-run -a python rank_latents.py`. After finishing, the results will be placed inside output/frame_comparisons/situational_similarity.


## Expected Results

* VAE similarity search, displaying the demonstrations of lowest (left) and highest similarity (right) based on the observation (top): ![523_good](https://github.com/user-attachments/assets/ba286e2a-c18b-4d4b-b77c-4c62cdfb656f)
* Screencapture of moment where agent (left) switches from navigation to execution mode based on demonstration (right) patch embeddings similarity: 
![Figure23](https://github.com/user-attachments/assets/120d52bb-8eae-419d-b548-f556e1cd7f6d)


