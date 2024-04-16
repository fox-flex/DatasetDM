# CUDA_LAUNCH_BLOCKING=4 
# COCO/ Instance Segmentation

# export PATH="/usr/local/cuda-12/bin:$PATH"
# export PYTHONPATH="$(pwd)"

CUDA_VISIBLE_DEVICES=1 python tools/train_sod_UHRSD.py \
    --save_name train_UHRSD_3 \
    --config ./config/UHRSD/UHRSD_50images.yaml \
    --image_limitation 10000 \
    --data_path ./data/full
    --dataset UHRSD