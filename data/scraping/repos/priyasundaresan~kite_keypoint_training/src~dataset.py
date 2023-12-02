import torch
import random
import cv2
import time
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import pickle
import os
from datetime import datetime
import clip
#import openai

# No domain randomization
transform = transforms.Compose([transforms.ToTensor()])

def normalize(x):
    return F.normalize(x, p=1)

def gauss_2d_batch(width, height, sigma, U, V, normalize_dist=False):
    U.unsqueeze_(1).unsqueeze_(2)
    V.unsqueeze_(1).unsqueeze_(2)
    X,Y = torch.meshgrid([torch.arange(0., width), torch.arange(0., height)])
    X,Y = torch.transpose(X, 0, 1).cuda(), torch.transpose(Y, 0, 1).cuda()
    G=torch.exp(-((X-U.float())**2+(Y-V.float())**2)/(2.0*sigma**2))
    if normalize_dist:
        return normalize(G).double()
    return G.double()

def bimodal_gauss(G1, G2, normalize=False):
    bimodal = torch.max(G1, G2)
    if normalize:
        return normalize(bimodal)
    return bimodal

def vis_gauss(gaussians, img, text):
    gaussians = gaussians.cpu().numpy()
    h = gaussians.squeeze()
    output = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX)
    output = np.stack((output,)*3, axis=-1)
    result = np.hstack((img, output))
    cv2.putText(result, text, (20,20), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255,255,255), 1, 2)
    cv2.imwrite('test.png', result)

class LanguageKeypointsDataset(Dataset):
    def __init__(self, num_keypoints, img_folder, kpts_folder, text_folder, img_height, img_width, transform, multimodal=False, gauss_sigma=8):
        #openai_api_key = "" 
        #openai.api_key = openai_api_key
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
        self.clip_model.cuda().eval()

        self.img_height = img_height
        self.img_width = img_width
        self.gauss_sigma = gauss_sigma
        self.multimodal = multimodal
        self.transform = transform

        self.imgs = []
        self.keypoints = []
        self.text_feats = np.zeros((0, 512), dtype=np.float32)

        self.texts = []
        for i in range(len(os.listdir(kpts_folder))):
            keypoints = np.load(os.path.join(kpts_folder, '%05d.npy'%i)).reshape(-1, 2)[:num_keypoints, :]
            for j in range(num_keypoints):
                keypoints[j,0] = np.clip(keypoints[j,0], 0, self.img_width-1)
            self.imgs.append(os.path.join(img_folder, '%05d.jpg'%i))
            self.keypoints.append(torch.from_numpy(keypoints).cuda())
            text = str(np.load(os.path.join(text_folder, '%05d.npy'%i))) + '.'
            self.texts.append(text)

        #text_tokens = clip.tokenize(texts).cuda() #tokenize
        #text_i = 0
        #while text_i < len(text_tokens):
        #    batch_size = min(len(text_tokens) - text_i, 512)
        #    text_batch = text_tokens[text_i:text_i+batch_size]
        #    with torch.no_grad():
        #        batch_feats = self.clip_model.encode_text(text_batch).float()
        #    batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        #    batch_feats = np.float32(batch_feats.cpu())
        #    self.text_feats = np.concatenate((self.text_feats, batch_feats), axis=0)
        #    text_i += batch_size

        #print(self.text_feats.shape)

    def __getitem__(self, index):  
        img_np = cv2.imread(self.imgs[index])
        img = self.transform(img_np)
        keypoints = self.keypoints[index]
        #text_feats = self.text_feats[index]
        text = self.texts[index]
        U = keypoints[:,0]
        V = keypoints[:,1]
        gaussians = gauss_2d_batch(self.img_width, self.img_height, self.gauss_sigma, U, V)
        if self.multimodal:
            mm_gauss = gaussians[0]
            for i in range(1, len(gaussians)):
                mm_gauss = bimodal_gauss(mm_gauss, gaussians[i])
            mm_gauss.unsqueeze_(0)
            gaussians = mm_gauss
        return img, gaussians, text, img_np
    
    def __len__(self):
        return len(self.keypoints)

if __name__ == '__main__':
    IMG_WIDTH = 240
    IMG_HEIGHT = 240
    GAUSS_SIGMA = 6
    dataset_name = 'dsetv0'
    dataset_dir = '/host/data/%s'%dataset_name
    test_dataset = LanguageKeypointsDataset('%s/images'%dataset_dir, '%s/keypoints'%dataset_dir, '%s/lang'%dataset_dir, IMG_HEIGHT, IMG_WIDTH, transform, multimodal=True, gauss_sigma=GAUSS_SIGMA)
    img, gaussians, text, img_np = test_dataset[0]
    print(img.shape, gaussians.shape, text)
    vis_gauss(gaussians, img_np, text)
