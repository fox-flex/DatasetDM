# CUDA_VISIBLE_DEVICES=3 
# Depth Estimation  KITTI
CUDA_VISIBLE_DEVICES=0 \
    python tools/parallel_generate_SOD_UHRSD.py \
    --sd_ckpt 'CompVis/stable-diffusion-v1-4' \
    --grounding_ckpt './checkpoint/train_UHRSD_4/latest_checkpoint.pth' \
    --n_each_class 10 \
    --outdir './data_gen/sod_UHRSD_4/' \
    --thread_num 1 \
    --H 512 \
    --W 512 \
    --config './config/UHRSD/UHRSD_50images.yaml' \
    --prompt_root "./dataset/Prompts_From_GPT/SOD"