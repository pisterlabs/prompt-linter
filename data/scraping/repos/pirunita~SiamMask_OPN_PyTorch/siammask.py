# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import argparse
import cv2
import imageio
import os
import glob

import torch.nn as nn
from Siammask_sharp.tools.test import *
from guidance import Guide

from OnionPeel.OPN import OPN
from OnionPeel.TCN import TCN
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--name', type=str, default='SiamMask')
parser.add_argument('--resume', default='./Siammask_sharp/SiamMask_DAVIS.pth', type=str,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='./Siammask_sharp/config/config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='./data/tennis', help='datasets')

parser.add_argument('--root_path', default='./results/')
parser.add_argument('--result_path', default='./results/tennis/')
parser.add_argument('--mask_path', default='./results/tennis/masks/')
parser.add_argument('--padding_factor', type=float, default=0.05, help='padding factor in guidance')

parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Make directory
    if not os.path.exists(args.root_path):
        os.makedirs(args.result_path)
        
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    
    if not os.path.exists(args.mask_path):
        os.makedirs(args.mask_path)
    
    # Setup SiamMask Model
    cfg = load_config(args)
    from Siammask_sharp.tools.custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Parse Image file
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]

    # Select ROI
    rect_img = ims[0].copy()
    guide = Guide(args.name, rect_img, args.padding_factor)
    
    
    x = guide.minX
    y = guide.minY
    w = guide.maxX - x
    h = guide.maxY - y
    toc = 0
    
    
    for f, im in enumerate(ims):
        opn_image = im.copy()
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        elif f > 0:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr
            for i in range(3):
                opn_image[:, :, i] = (mask > 0) * 255
            
            cv2.imshow(args.name, opn_image)
            cv2.imwrite(os.path.join(args.mask_path, '{:05d}.png'.format(f)), cv2.cvtColor(opn_image, cv2.COLOR_BGR2GRAY))
            key = cv2.waitKey(1)
            if key > 0:
                break
    
        
        toc += cv2.getTickCount() - tic
    
    siammask.cpu()
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
