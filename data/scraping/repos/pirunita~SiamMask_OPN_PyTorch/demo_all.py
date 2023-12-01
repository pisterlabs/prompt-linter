# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import argparse
import cv2
import glob
import imageio
import os


import torch.nn as nn
from Siammask_sharp.tools.test import *
from guidance import Guide

from OnionPeel.OPN import OPN
from OnionPeel.TCN import TCN
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--name', type=str, default='SiamMask')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='./Siammask_sharp/config/config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='./data/tennis', help='datasets')

parser.add_argument('--save_path', default='./results/')
parser.add_argument('--mask_path', default='./results/masks/')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Make directory
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
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
    
    """
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        x, y, w, h = init_rect
    except:
        exit()
    """
    rect_img = ims[0].copy()
    guide = Guide(args.name, rect_img)
    masks = []
    
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
            #opn_image = cv2.cvtColor(opn_image, cv2.COLOR_BGR2GRAY)
            #im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            #cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imshow(args.name, opn_image)
            #cv2.imwrite(os.path.join(args.mask_path, '{:05d}.png'.format(f)), opn_image)
            key = cv2.waitKey(1)
            if key > 0:
                break
    
        masks.append(cv2.cvtColor(opn_image, cv2.COLOR_BGR2GRAY))
        toc += cv2.getTickCount() - tic
    
    #imageio.mimsave(os.path.join(args.mask_path, 'mask.gif'), masks)
    siammask.cpu()
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))

    MEM_EVERY = 5
    T = len(ims)
    H, W = 240, 424
    frames = np.empty((T, H, W, 3), dtype=np.float32)
    holes = np.empty((T, H, W, 1), dtype=np.float32)
    dists = np.empty((T, H, W, 1), dtype=np.float32)
    
    ims = [Image.open(imf).convert('RGB') for imf in img_files] 
    
    for i in range(T):
        raw_frame = np.array(ims[i]) / 255
        raw_frame = cv2.resize(raw_frame, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
        
        raw_mask = np.array(Image.fromarray(masks[i], 'P'), dtype=np.uint8)
        raw_mask = (raw_mask > 0.5).astype(np.uint8)
        raw_mask = cv2.resize(raw_mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
        raw_mask = cv2.dilate(raw_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)))
        
        
        frames[i] = raw_frame
        holes[i, :, :, 0] = raw_mask.astype(np.float32)
        dists[i, :, :, 0] = cv2.distanceTransform(raw_mask, cv2.DIST_L2, maskSize=5)
        
    cv2.destroyAllWindows()
    frames = torch.from_numpy(np.transpose(frames, (3, 0, 1, 2)).copy()).float()
    holes = torch.from_numpy(np.transpose(holes, (3, 0, 1, 2)).copy()).float()
    dists = torch.from_numpy(np.transpose(dists, (3, 0, 1, 2)).copy()).float()
    
    # remove hole
    frames = frames * (1-holes) + holes*torch.tensor([0.485, 0.456, 0.406]).view(3,1,1,1)
    # valids area
    valids = 1-holes
    # unsqueeze to batch 1
    frames = frames.unsqueeze(0)
    holes = holes.unsqueeze(0)
    dists = dists.unsqueeze(0)
    valids = valids.unsqueeze(0)
    
    MEM_EVERY = 5
    comps = torch.zeros_like(frames)
    ppeds = torch.zeros_like(frames)
    
    midx = list(range(0, T, MEM_EVERY))
    with torch.no_grad():
        mkey, mval, mhol = opn(frames[:,:,midx], valids[:,:,midx], dists[:,:,midx])
    
    for f in range(T):
        if f in midx:
            ridx = [i for i in range(len(midx)) if i != int(f/MEM_EVERY)]
        else:
            ridx = list(range(len(midx)))
        
        fkey, fval, fhol = mkey[:, :, ridx], mval[:, :, ridx], mhol[:, :, ridx]
        
        for r in range(999):
            if r == 0:
                comp = frames[:, :, f]
                dist = dists[:, :, f]
            with torch.no_grad():
                comp, dist = opn(fkey, fval, fhol, comp, valids[:, :, f], dist)
            
            comp, dist = comp.detach(), dist.detach()
            if torch.sum(dist).item() == 0:
                break
        
        comps[:, :, f] = comp
    
    ppeds[:, :, 0] = comps[:, :, 0]
    hidden = None
    for f in range(T):
        with torch.no_grad():
            pped, hidden =\
                tcn(ppeds[:, :, f-1], holes[:, :, f-1], comps[:, :, f], holes[:, :, f], hidden)

            ppeds[:, :, f] = pped
    
    # visualize
    
    for f in range(T):
        est = (ppeds[0,:,f].permute(1,2,0).detach().cpu().numpy() * 255.).astype(np.uint8)
        true = (frames[0,:,f].permute(1,2,0).detach().cpu().numpy() * 255.).astype(np.uint8) # h,w,3
        mask = (dists[0,0,f].detach().cpu().numpy() > 0).astype(np.uint8) # h,w,1
        
        
        canvas = np.concatenate([true, est], axis=0)
        canvas = Image.fromarray(canvas)
        canvas.save(os.path.join(args.save_path, '{:05d}.jpg'.format(f)))
        
    
    
        
    
        

    
    