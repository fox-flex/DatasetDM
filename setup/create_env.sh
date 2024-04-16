#!/bin/bash

set -e

conda create -y python=3.10 --name dm
conda activate dm
pip install --upgrade pip
pip install -r requirements.txt
pip install \
    torch==2.2.0 \
    torchaudio==2.2.0 \
    torchgeometry==0.1.2 \
    torchmetrics==0.11.4 \
    torchvision==0.17.0 \
    open-clip-torch==2.20.0 \
    clip-anytorch==2.5.2 \
    multilingual-clip==1.0.10 \
    pytorch-lightning==2.2.2 \
    transformers==4.39.3 \
    tokenizers==0.15.2 \
    sentence_transformers==2.6.1 \
    xformers==0.0.24 \
    diffusers==0.20.2 \
    accelerate==0.27.2 \
    mmcv==1.7.1 \
    mmdet==3.2.0 \
    mmengine==0.10.3 \
    mmpose==1.2.0 \
    mmcv==1.7.1 \
    tensorboard==2.12.1 \
    tensorboard-data-server==0.7.0 \
    tensorboard-plugin-wit==1.8.1 \
    safetensors==0.4.2 \
    timm==0.9.16

pip install git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1#egg=clip
pip install git+https://github.com/facebookresearch/detectron2.git
python -c "import detectron2; print(detectron2.__version__)"

