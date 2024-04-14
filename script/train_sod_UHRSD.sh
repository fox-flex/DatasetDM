# CUDA_LAUNCH_BLOCKING=4 
# COCO/ Instance Segmentation

CUDA_VISIBLE_DEVICES=1 python tools/train_sod_UHRSD.py \
    --save_name Train_200_images_t1_attention_transformer_UHRSD_10layer \
    --config ./config/UHRSD/UHRSD_50images.yaml \
    --image_limitation 200 \
    --dataset UHRSD