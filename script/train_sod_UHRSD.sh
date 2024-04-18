# CUDA_LAUNCH_BLOCKING=4 
# COCO/ Instance Segmentation

# export PATH="/usr/local/cuda-12/bin:$PATH"
# export PYTHONPATH="$(pwd)"

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0
python tools/train_sod_UHRSD.py \
    --save_name train_UHRSD_5 \
    --config ./config/UHRSD/UHRSD_50images.yaml \
    --image_limitation 10000 \
    --data_path ./data/full \
    --dataset UHRSD \
    --start_ckpt ./checkpoint/train_UHRSD_4/latest_checkpoint.pth