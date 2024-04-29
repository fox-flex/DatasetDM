from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
# from diffusers import StableDiffusionPipeline
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
import os
from pathlib import Path
import argparse
from IPython.display import Image, display
from pytorch_lightning import seed_everything
from tqdm import tqdm
from dataset import UHRSD
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from model.unet import UNet2D,get_feature_dic,clear_feature_dic
from model.depth_module import Depthmodule
import yaml
from get_prompts_coco import get_prompts

# from tools.train_instance_coco import dict2obj
class dict2obj(object):
    def __init__(self, d):
        self.__dict__['d'] = d
 
    def __getattr__(self, key):
        value = self.__dict__['d'][key]
        if type(value) == type({}):
            return dict2obj(value)
 
        return value

import torch.optim as optim
from train import AttentionStore
import torch.nn.functional as F
from scipy.special import softmax


# from detectron2.modeling.postprocessing import sem_seg_postprocess
# from detectron2.utils.memory import retry_if_cuda_oom
# from torch import autocast
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from random import choice
# classes = {
#                 0: 'road',
#                 1: 'sidewalk',
#                 2: 'building',
#                 3: 'wall',
#                 4: 'fence',
#                 5: 'pole',
#                 6: 'traffic light',
#                 7: 'traffic sign',
#                 8: 'vegetation',
#                 9: 'terrain',
#                 10: 'sky',
#                 11: 'person',
#                 12: 'rider',
#                 13: 'car',
#                 14: 'truck',
#                 15: 'bus',
#                 16: 'train',
#                 17: 'motorcycle',
#                 18: 'bicycle'
#             }





def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def freeze_params(params):
    for param in params:
        param.requires_grad = False
        
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def plot_mask(img, masks, colors=None, alpha=0.8,indexlist=[0,1]) -> np.ndarray:
    """Visualize segmentation mask.

    Parameters
    ----------
    img: numpy.ndarray
        Image with shape `(H, W, 3)`.
    masks: numpy.ndarray
        Binary images with shape `(N, H, W)`.
    colors: numpy.ndarray
        corlor for mask, shape `(N, 3)`.
        if None, generate random color for mask
    alpha: float, optional, default 0.5
        Transparency of plotted mask

    Returns
    -------
    numpy.ndarray
        The image plotted with segmentation masks, shape `(H, W, 3)`

    """
    H,W= masks.shape[0],masks.shape[1]
    color_list=[[255,97,0],[128,42,42],[220,220,220],[255,153,18],[56,94,15],[127,255,212],[210,180,140],[221,160,221],[255,0,0],[255,128,0],[255,255,0],[128,255,0],[0,255,0],[0,255,128],[0,255,255],[0,128,255],[0,0,255],[128,0,255],[255,0,255],[255,0,128]]*6
    final_color_list=[np.array([[i]*512]*512) for i in color_list]
    
    background=np.ones(img.shape)*255
    count=0
    colors=final_color_list[indexlist[count]]
    for mask, color in zip(masks, colors):
        color=final_color_list[indexlist[count]]
        mask = np.stack([mask, mask, mask], -1)
        img = np.where(mask, img * (1 - alpha) + color * alpha,background*0.4+img*0.6 )
        count+=1
    return img.astype(np.uint8)

def aggregate_attention(attention_store, res: int, from_where: List[str], is_cross: bool, select: int, prompts=None):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
#                 print(item.reshape(len(prompts), -1, res, res, item.shape[-1]).shape)
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[0]
                out.append(cross_maps)

    out = torch.cat(out, dim=0)
    return out

def sub_processor(pid , opt):
    
    f = open(opt.config)
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = dict2obj(cfg)
    
    torch.cuda.set_device(pid)
    text = 'processor %d' % pid
    print(text)
    
    seed_everything(opt.seed)
    

    task = "depth"
    with open('./config/token.txt', 'r') as f:
        MY_TOKEN = f.read().strip()
    LOW_RESOURCE = False 
    NUM_DIFFUSION_STEPS = 50
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ckpts_path = "./dataset/ckpts/imagenet/"
    tokenizer = CLIPTokenizer.from_pretrained(ckpts_path, subfolder="tokenizer")
    
    #VAE
    vae = AutoencoderKL.from_pretrained(ckpts_path, subfolder="vae")
    freeze_params(vae.parameters())
    vae=vae.to(device)
    vae.eval()
    
    #UNet2DConditionModel UNet2D
    unet = UNet2D.from_pretrained(ckpts_path, subfolder="unet")
    freeze_params(unet.parameters())
    unet=unet.to(device)
    unet.eval()
    
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(ckpts_path, "text_encoder"))
    freeze_params(text_encoder.parameters())
    text_encoder=text_encoder.to(device)
    text_encoder.eval()
    
    scheduler = StableDiffusionPipeline.from_pretrained(opt.sd_ckpt, use_auth_token=MY_TOKEN).to(device).scheduler
    
    depth_module=Depthmodule(max_depth=cfg.Depth_Decorder.max_depth).to(device)
    
    print('load weight:',opt.grounding_ckpt)
    base_weights = torch.load(opt.grounding_ckpt, map_location="cpu")
    # try:
    depth_module.load_state_dict(base_weights, strict=True)
    # except:
    #     new_state_dict = OrderedDict()
    #     for k, v in base_weights.items():
    #         name = k[7:]   # remove `vgg.`
    #         new_state_dict[name] = v 
    #     seg_module.load_state_dict(new_state_dict, strict=True)
        

    Path(opt.outdir).mkdir(parents=True, exist_ok=True)
    outpath = opt.outdir
    
    Image_path = os.path.join(outpath, "image")
    os.makedirs(Image_path, exist_ok=True)
        
    Mask_path = os.path.join(outpath, "mask")
    os.makedirs(Mask_path, exist_ok=True)
    
    batch_size = opt.n_samples

    controller = AttentionStore()
    ptp_utils.register_attention_control(unet, controller)
    
    all_prompts = list(get_prompts(opt.prompt_root))
    number_per_thread_num = int(int(opt.n_each_class)/opt.thread_num)
    
    # sub_classes_list = []
    # #read general prompt txt 
    # prompt_path = os.path.join(opt.prompt_root,"general.txt")
    # with open(prompt_path, "r") as f2:
    #     sub_classes_list = [line.strip() for line in f2.readlines()]
    
    refiner = ptp_utils.Refiner(device=device)
    get_num_dif = lambda x: int(np.ceil(np.log10(x)))
    total_class_num = get_num_dif(len(all_prompts))
    total_seed_num = get_num_dif(number_per_thread_num)


    with torch.no_grad():
        seed = 0
        for sub_class_id, sub_classes_list in enumerate(all_prompts):

            print(f"prompt candidates num: {len(sub_classes_list)}")

            pbar = tqdm(range(number_per_thread_num), total=number_per_thread_num, desc=f"Thread {pid}")
            # pbar = range(number_per_thread_num)
            for n in pbar:
                out_name = f'{pid}_{sub_class_id:0{total_class_num}d}_{seed:0{total_seed_num}d}'

                # clear all features and attention maps
                clear_feature_dic()
                controller.reset()


                g_cpu = torch.Generator().manual_seed(seed)

                prompts = [choice(sub_classes_list)]
                prompt_size = 50 
                # pbar.set_description(f"Prompt: {prompts[0][:prompt_size]}{'...' if len(prompts[0]) > prompt_size else ''}")
                # print(prompts[0])
                start_code = None
                if opt.fixed_code:
                    print('start_code')
                    start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)


                if isinstance(prompts, tuple):
                    prompts = list(prompts)

                images_here, x_t, _ = ptp_utils.text2image(unet,vae,tokenizer,text_encoder,scheduler, prompts, controller,  num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=5, generator=g_cpu, low_resource=LOW_RESOURCE, Train=False)
                images_here = ptp_utils.save_images(images_here, out_put = os.path.join(Image_path, f'{out_name}.jpg'))

                # depth
                diffusion_features=get_feature_dic()
                pred_depth=depth_module(diffusion_features,controller,prompts,tokenizer)
                pred_depth = np.array(pred_depth[0][0].cpu()).astype('float32') / cfg.Depth_Decorder.max_depth * 256 

                pred_depth = refiner.refine(images_here, pred_depth)

                cv2.imwrite(os.path.join(Mask_path, f'{out_name}.png'), pred_depth)
                seed+=1

            pbar.close()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a photo of a lion on a mountain top at sunset",
        help="the prompt to render"
    )
    parser.add_argument(
        "--category",
        type=str,
        nargs="?",
        default="lion",
        help="the category to ground"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./DataDiffusion/VOC/"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--thread_num",
        type=int,
        default=8,
        help="number of threads",
    )
    
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--prompt_root",
        type=str,
        help="uses prompt",
        default="./dataset/Prompts_From_GPT/cityscapes"
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_each_class",
        type=int,
        default=20,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--sd_ckpt",
        type=str,
        default="stable_diffusion.ckpt",
        help="path to checkpoint of stable diffusion model",
    )
    parser.add_argument(
        "--grounding_ckpt",
        type=str,
        default="grounding_module.pth",
        help="path to checkpoint of grounding module",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()
    return opt

def main():
    opt = parse_args()
    
    import multiprocessing as mp
    import threading
    result_dict = mp.Manager().dict()
    mp = mp.get_context("spawn")
    processes = []
#     per_thread_video_num = int(len(coco_category_list)/thread_num)
#     thread_num=8
    print('Start Generation')
    for i in range(opt.thread_num):
#         if i == thread_num - 1:
#             sub_video_list = coco_category_list[i * per_thread_video_num:]
#         else:
#             sub_video_list = coco_category_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]

        p = mp.Process(target=sub_processor, args=(i, opt))
        p.start()
        processes.append(p)


    for p in processes:
        p.join()

    result_dict = dict(result_dict)


if __name__ == "__main__":
    main()
