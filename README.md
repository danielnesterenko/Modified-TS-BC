# Modified TS-BC
Modified TS-BC is a modification of the Targeted Search-Based Behavioral Cloning model for playing Minecraft.

## Disclaimer
This project is a modification of [TS-BC]([https://pages.github.com/]https://github.com/JulianBvW/TS-BC).

It utilizes parts of the following projects:
* STEVE-1: https://github.com/Shalev-Lifshitz/STEVE-1
* DepthAnything: https://github.com/LiheYoung/Depth-Anything
* DINOv2: https://github.com/facebookresearch/dinov2

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
* Download 'attn.pth' from https://drive.google.com/file/d/1uaZM1ZLBz2dZWcn85rZmjP7LV6Sg5PZW/view and move into weights/mineclip
* Download 'steve1_prior.pt' from https://drive.google.com/uc?id=1OdX5wiybK8jALVfP5_dEo0CWm9BQbDES and move into weights/cvae
* Download 'model_vid10000_4x4x8.pth' from https://drive.google.com/uc?export=download&id=1588qzaRGNvQWsibVBDdp_9zA8VFHd4wE and move into weights/pretrained_vae
* Download the directory 'data' from https://drive.google.com/drive/folders/1WdzwLzmUilVKSt6sYVvSZQLoy2FPHhk0?usp=sharing and move it into dataset/
* Download the directory 'facebookresearch_dinov2_main' from https://drive.google.com/drive/folders/1lp5mSZWSDg0Ca7ebJmujK5YdH-NDSgZc?usp=sharing and move it into torchhub/

## Indexing the Embedding Space

* Run train.py with the new conda environment enabled.
