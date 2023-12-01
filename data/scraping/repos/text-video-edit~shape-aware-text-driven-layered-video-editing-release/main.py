import sys, os, argparse, time, json, yaml
import numpy as np
import torch
from path import Path
from tqdm import tqdm

import imageio
import imageio.v3 as iio

from editor import Editor

from train import *
from guidance import StableDiffusion

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('edit_dir', type=str)
    
    parser.add_argument('--height', type=int, default=432)
    parser.add_argument('--width', type=int, default=768)

    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --dir_text")
    parser.add_argument('-O2', action='store_true', help="equals --fp16 --dir_text")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--eval_interval', type=int, default=10, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--guidance', type=str, default='stable-diffusion', help='choose from [stable-diffusion, clip]')
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--edit_bg', type=str, default=None)
    parser.add_argument('--init_video', action='store_true')

    ### training options
    parser.add_argument('--iters', type=int, default=100, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='scratch')
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")

    parser.add_argument('--lambda_text_guidance', type=float, default=1)
    parser.add_argument('--lambda_text_guidance_scale', type=float, default=100)
    parser.add_argument('--lambda_target_pixel', type=float, default=1e6)
    parser.add_argument('--lambda_target_perceptual', type=float, default=0)
    parser.add_argument('--lambda_texture_pixel', type=float, default=100)
    parser.add_argument('--lambda_delta_pixel', type=float, default=1)
    parser.add_argument('--lambda_texture_smooth', type=float, default=1000)
    parser.add_argument('--lambda_mask_smooth', type=float, default=1000)
    parser.add_argument('--texture_smooth_scales', type=float, nargs='+', default=[0])
    parser.add_argument('--lambda_delta_smooth', type=float, default=1)
    parser.add_argument('--lambda_corr_smooth', type=float, default=10)
    parser.add_argument('--lambda_corr_mask', type=float, default=1000)
    parser.add_argument('--sd_min_step', type=float, default=2)
    parser.add_argument('--sd_max_step', type=float, default=998)

    parser.add_argument('--interpolate_fg', action='store_true', default=False)
    parser.add_argument('--interpolate_bg', action='store_true', default=False)

     
    opt = parser.parse_args()
    

    seed_everything(opt.seed)


    model = Editor(opt).to(device)
    setattr(opt, 'text', model.text_prompt)
    
    if opt.test:
        guidance = None # no need to load guidance model at test

        trainer = Trainer('df', opt, model, guidance, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)
        trainer.test(full_demo=True)
            
    else:
        max_epoch = opt.iters

        optimizer = lambda model: torch.optim.AdamW(model.get_params(opt.lr), weight_decay=0)
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: max(0.0, 10 ** (-iter*0.0002)))
        
        guidance = StableDiffusion(device, False, opt.sd_min_step, opt.sd_max_step)
        
        trainer = Trainer('df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True)
        
        trainer.train(max_epoch)

        # also test
        trainer.test()
