# CUDA_VISIBLE_DEVICES=3 
# Pose Estimation  COCO
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python tools/parallel_generate_Pose_COCO_HeatMap.py --sd_ckpt './models/ldm/stable-diffusion-v1/stable_diffusion.ckpt' --grounding_ckpt './checkpoint/Train_800_images_t1_attention_transformer_Pose_COCO_HeatMap_10layers/latest_checkpoint.pth' --number_data 20000 --outdir './DataDiffusion/COCO_Pose_Train_800_images_t1/' --thread_num 7 --H 512 --W 512 --config './config/coco_pose/coco_pose.yaml' --prompt_root "./dataset/Prompts_From_GPT/coco_pose"