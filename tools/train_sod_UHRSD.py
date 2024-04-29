import sys
from copy import deepcopy
from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import cv2
import json
import torchvision
import argparse
import multiprocessing as mp
import torch.nn as nn
import threading
from random import choice
import random
import os
from distutils.version import LooseVersion
import argparse
from IPython.display import display
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm
from dataset import UHRSD
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline
from model.diffusers.models.unet_2d_condition import UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from model.unet import UNet2D,get_feature_dic,clear_feature_dic
from model.depth_module import Depthmodule
import torch.optim as optim
import torch.nn.functional as F
# from model.segment.criterion import SetCriterion
# from model.segment.matcher import HungarianMatcher
# from detectron2.modeling.postprocessing import sem_seg_postprocess
# from detectron2.utils.memory import retry_if_cuda_oom
import yaml
# from tools.utils import mask_image
from torch.optim.lr_scheduler import StepLR
# from detectron2.structures import Boxes, ImageList, Instances, BitMasks

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

LOW_RESOURCE = False


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            if self.activate:
                self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        

class dict2obj(object):
    def __init__(self, d):
        self.__dict__['d'] = d
 
    def __getattr__(self, key):
        value = self.__dict__['d'][key]
        if type(value) == type({}):
            return dict2obj(value)
 
        return value

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if self.activate:
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
    #         if attn.shape[1] <= 128 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if self.activate:
            if len(self.attention_store) == 0:
                self.attention_store = self.step_store
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i] += self.step_store[key][i]
            self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.activate = True
        
def freeze_params(params):
    for param in params:
        param.requires_grad = False
        


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss


class BCE_Dice_Loss(nn.Module):
    def __init__(self, reduction=None):
        if reduction is None:
            self.reduction_fn = lambda x: x
        elif reduction == 'mean':
            self.reduction_fn = torch.mean
        elif reduction == 'sum':
            self.reduction_fn = torch.sum
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
        super(BCE_Dice_Loss, self).__init__()

    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target)
        inter = (pred * target).sum()
        eps = 1e-5
        dice = 1 - (2 * inter + eps) / (pred.sum() + target.sum() + eps)
        loss = 5 * bce + 5*dice

        loss = self.reduction_fn(loss)
        return loss

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        default="./config/",
        help="config for training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--start_ckpt",
        type=str,
        default="",
        help="ckpt to load for starting model (optional)",
    )
    parser.add_argument(
        "--image_limitation",
        type=int,
        default=5,
        help="image_limitation",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/UHRSD_TR",
        help="data_path",
    )
    parser.add_argument(
        "--dataset", type=str, default="Cityscapes", help="dataset: VOC/Cityscapes/DepthCut"
    )
    parser.add_argument(
        "--save_name",
        type=str,
        help="the save dir name",
        default="Test"
    )
    opt = parser.parse_args()
    return opt

def main():
    opt = parse_args()
    seed_everything(opt.seed)
    
    f = open(opt.config)
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = dict2obj(cfg)
    
    opt.batch_size = cfg.DATASETS.batch_size
    
    # dataset
    dataset = UHRSD(
        data_path = opt.data_path,
        image_limitation = opt.image_limitation
    )
    # loss_fn = SiLogLoss()
    # loss_fn = nn.BCELoss(reduction='mean')
    loss_fn = BCE_Dice_Loss(reduction='mean')
    
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    print('***********************   begin   **********************************')
    save_dir = 'checkpoint'
    os.makedirs(save_dir, exist_ok=True)
    learning_rate = cfg.SOLVER.learning_rate
    adam_weight_decay = cfg.SOLVER.adam_weight_decay
    total_epoch = cfg.SOLVER.total_epoch
    
    ckpt_dir = os.path.join(save_dir, opt.save_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # ckpts_path = "./dataset/ckpts/imagenet/"
    # ckpts_path = '~/.cache/huggingface/hub/models--SG161222--RealVisXL_V3.0/snapshots/4a3f0e44d3abcc0c3ee48fe85e337d78075d1445/'
    ckpts_path = '/home/ubuntu-system/.cache/huggingface/hub/models--SG161222--RealVisXL_V3.0/snapshots/4a3f0e44d3abcc0c3ee48fe85e337d78075d1445'
    tokenizer = CLIPTokenizer.from_pretrained(ckpts_path, subfolder="tokenizer")
    
    #VAE
    vae = AutoencoderKL.from_pretrained(ckpts_path, subfolder="vae")
    freeze_params(vae.parameters())
    vae=vae.to(device)
    vae.eval()
    
    unet = UNet2D.from_pretrained(ckpts_path, subfolder="unet")
    freeze_params(unet.parameters())
    unet=unet.to(device)
    unet.eval()
    
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(ckpts_path, "text_encoder"))
    freeze_params(text_encoder.parameters())
    text_encoder=text_encoder.to(device)
    text_encoder.eval()
    

    mask_module=Depthmodule(max_depth=cfg.Depth_Decorder.max_depth).to(device)
    
    # if opt.start_ckpt:
    #     print(f"loading model from{opt.start_ckpt}")
    #     base_weights = torch.load(opt.start_ckpt, map_location="cpu")
    #     mask_module.load_state_dict(base_weights, strict=True)

    noise_scheduler = DDPMScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    
    os.makedirs(os.path.join(ckpt_dir, 'training'), exist_ok=True)
    print("learning_rate:",learning_rate)
    g_optim = optim.Adam(
            [{"params": mask_module.parameters()},],
            lr=learning_rate
          )
    scheduler = StepLR(g_optim, step_size=2, gamma=0.1)
    
    
    start_code = None
    
    LOW_RESOURCE = cfg.Diffusion.LOW_RESOURCE
    NUM_DIFFUSION_STEPS = cfg.Diffusion.NUM_DIFFUSION_STEPS
    GUIDANCE_SCALE = cfg.Diffusion.GUIDANCE_SCALE
    MAX_NUM_WORDS = cfg.Diffusion.MAX_NUM_WORDS
    
    
    controller = AttentionStore()
    ptp_utils.register_attention_control(unet, controller)
    
    for j in range(1, total_epoch+1):
        # print('Epoch ' +  str(j) + '/' + str(total_epoch))
        # pbar = tqdm(dataloader)
        
        sz = np.ceil(np.log10(total_epoch)).astype(int)
        ep_str = f'Epoch {j:0{sz}d}/{total_epoch}'
        pbar = tqdm(enumerate(dataloader), desc=ep_str, total=len(dataloader))
        
        for step, batch in pbar:
            do_vis = step%500 == 1
            g_cpu = torch.Generator().manual_seed(random.randint(1, 10000000))
            
            # clear all features and attention maps
            clear_feature_dic()
            controller.reset()
            
            
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            prompts = batch["prompt"]
            original_mask = batch["original_mask"]
            original_image = batch["original_image"]
            cache_batch = batch["cache"]
            if True:
                assert len(cache_batch) == 1, 'supported batch_size==1 only'
                cache = cache_batch[0]
                batch_size = image.shape[0]
                latents = vae.encode(image.to(device)).latent_dist.sample().detach()
                latents = latents * 0.18215
                
                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                
                # set timesteps
                noise_scheduler.set_timesteps(NUM_DIFFUSION_STEPS)
                stepss = noise_scheduler.timesteps[-1]
                timesteps = torch.ones_like(timesteps) * stepss
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                start_code = noisy_latents.to(latents.device)
        
                
                images_here, x_t, vae_features = ptp_utils.text2image(
                    unet,vae,tokenizer,text_encoder,
                    noise_scheduler, prompts, controller,
                    latent=start_code,
                    num_inference_steps=NUM_DIFFUSION_STEPS,
                    guidance_scale=5, generator=g_cpu,
                    low_resource=LOW_RESOURCE, Train=True
                )

                if do_vis:
                    prefix = f'ep_{j:05d}_step_{step:05d}'

                    ptp_utils.save_images(images_here, out_put = (os.path.join(ckpt_dir,  'training', f'{prefix}_viz_sample.png')))

                    Image.fromarray(original_mask.cpu().numpy()[0].astype(np.uint8) * 255).save(os.path.join(ckpt_dir, 'training', f'{prefix}_original_mask.png'))
                    Image.fromarray(original_image.cpu().numpy()[0].astype(np.uint8)).save(os.path.join(ckpt_dir, 'training', f'{prefix}_original_image.png'))
                
                diffusion_features=get_feature_dic()

                # # for cache, diffusion_feature in zip(cache_batch, diffusion_features):
                # to_cache = deepcopy(diffusion_features)
                # for k, v in to_cache.items():
                #     to_cache[k] = [v0.squeeze(0) for v0 in v]

                # with open(cache, 'wb') as cache_out:
                #     pickle.dump(to_cache, cache_out, -1)

            elif isinstance(cache_batch, dict):
                
                diffusion_features = cache_batch
            else:
                raise 1
                    
            # train segmentation
            diffusion_features = diffusion_features | vae_features
            pred_mask=mask_module(diffusion_features,controller,prompts,tokenizer)
            
            loss = []
            for b_index in range(batch_size):
                train_class_index=0
                pred_mask=torch.unsqueeze(pred_mask[b_index,train_class_index,:,:],0).unsqueeze(0)
                mask=mask[b_index].float().unsqueeze(0).unsqueeze(0).cuda()

                loss.append(loss_fn(pred_mask, mask))
            
            if len(loss)==0:
                pass
            else:
                total_loss=0
                for i in range(len(loss)):
                    total_loss+=loss[i]
                total_loss/=batch_size
                g_optim.zero_grad()
                total_loss.backward()
                g_optim.step()
        
        
            prompt_size = 22
            prompt = prompts[0][:prompt_size] + '...' if prompt_size < len(prompts[0]) else ''
            lr = float(g_optim.state_dict()['param_groups'][0]['lr'])
            pbar.set_description(f"{ep_str}: loss: {total_loss:0.6f}, "
                                 f"lr: {lr:0.6f}, prompt: {prompt:<{prompt_size+3}}")
            
            if do_vis:
                annotation_pred_gt = mask[0][0].cpu()
                pred_mask = pred_mask[0][0].cpu()
                annotation_pred_gt = annotation_pred_gt/annotation_pred_gt.max()*255
                pred_mask = pred_mask/pred_mask.max()*255
                viz_tensor2 = torch.cat([annotation_pred_gt, pred_mask], axis=1)

                torchvision.utils.save_image(
                    viz_tensor2,
                    os.path.join(ckpt_dir, 'training', f'{prefix}_viz_sample_seg.png')
                )
                    

        print("Saving latest checkpoint to",ckpt_dir)
        torch.save(mask_module.state_dict(), os.path.join(ckpt_dir, 'latest_checkpoint.pth'))
        if j%2==0:
            torch.save(mask_module.state_dict(), os.path.join(ckpt_dir, 'checkpoint_'+str(j)+'.pth'))
        scheduler.step()


if __name__ == "__main__":
    main()