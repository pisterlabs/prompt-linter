#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:10:55 2023

@author: shannon

I am trying my best to reconstruct the tabular diffusion, tab-ddpm,
to allow for multi output tabular diffusion


Code borrowed from https://www.kaggle.com/code/grishasizov/simple-denoising-diffusion-model-toy-1d-example
"""

import numpy as np
import json
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score

import sklearn.preprocessing as PP




def timestep_embedding(timesteps, dim, max_period=10000, device=torch.device('cuda:0')):
    """
    From https://github.com/rotot0/tab-ddpm
    
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1,device=device)
    return embedding

def generate_performance_weights(num_samples,num_metrics, gen_type='random'):
    
    weights = np.zeros((num_samples,num_metrics))
     
    if gen_type == 'random':
        for i in range(0,num_samples):
            a = np.random.rand(1,num_metrics)
            weights[i] = a/np.sum(a)
            
    elif gen_type == 'uniform':
        samples = []
        
        steps = np.linspace(0.0,1.0,11)
        
        for i in range(0, len(steps)):
            for j in range(0,len(steps)-i):
                samples.append([steps[i],steps[j],1.0-steps[i]-steps[j]])
        samples = np.array(samples)
        
        L = len(samples)
        
        print(L)
        
        A = np.random.randint(0,L,num_samples)
        
        for i in range(0,num_samples):
            weights[i] = samples[A[i]]

    return weights



# Now lets make a Denoise Model:
    
class Denoise_MLP_Model(torch.nn.Module):
    def __init__(self, DDPM_Dict, device=torch.device('cuda:0')):
        nn.Module.__init__(self)
        
        self.xdim = DDPM_Dict['xdim']
        self.ydim = DDPM_Dict['ydim']
        self.cdim = DDPM_Dict['cdim']
        self.tdim  = DDPM_Dict['tdim']
        self.net = DDPM_Dict['net']
        self.device = device
        
        self.fc = nn.ModuleList()
        
        self.fc.append(self.LinLayer(self.tdim,self.net[0]))
        
        for i in range(1, len(self.net)):
            self.fc.append(self.LinLayer(self.net[i-1],self.net[i]))
            
        self.fc.append(self.LinLayer(self.net[-1], self.tdim))
        
        self.fc.append(nn.Sequential(nn.Linear(self.tdim, self.xdim)))
        
    
        self.X_embed = nn.Linear(self.xdim, self.tdim)
        
        '''
        self.Y_embed = nn.Sequential(
            nn.Linear(self.ydim, self.tdim),
            nn.SiLU(),
            nn.Linear(self.tdim, self.tdim))
        '''
        self.Con_embed = nn.Sequential(
                    nn.Linear(self.cdim, self.tdim),
                    nn.SiLU(),
                    nn.Linear(self.tdim, self.tdim))
        
        
        self.time_embed = nn.Sequential(
            nn.Linear(self.tdim, self.tdim),
            nn.SiLU(),
            nn.Linear(self.tdim, self.tdim))
       
        
        
    def LinLayer(self, dimi, dimo):
        
        return nn.Sequential(nn.Linear(dimi,dimo),
                             nn.SiLU(),
                             nn.LayerNorm(dimo),
                             nn.Dropout(p=0.1))
        

    def forward(self, x, timesteps):
        a = self.X_embed(x)
        #print(a.dtype)
        x = a +  self.time_embed(timestep_embedding(timesteps, self.tdim))
        
        for i in range(0,len(self.fc)):
            x = self.fc[i](x)
    

        return x
    
class Denoise_ResNet_Model(torch.nn.Module):
    def __init__(self, DDPM_Dict):
        nn.Module.__init__(self)
        
        self.xdim = DDPM_Dict['xdim']
        self.ydim = DDPM_Dict['ydim']
        self.tdim  = DDPM_Dict['tdim']
        self.cdim = DDPM_Dict['cdim']
        self.net = DDPM_Dict['net']
        
        self.fc = nn.ModuleList()
        
        self.fc.append(self.LinLayer(self.tdim,self.net[0]))
        
        for i in range(1, len(self.net)):
            self.fc.append(self.LinLayer(self.net[i-1],self.net[i]))
            
        self.fc.append(self.LinLayer(self.net[-1], self.tdim))
        
        
        self.finalLayer = nn.Sequential(nn.Linear(self.tdim, self.xdim))
        
    
        self.X_embed = nn.Linear(self.xdim, self.tdim)
        
        '''
        self.Y_embed = nn.Sequential(
            nn.Linear(self.ydim, self.tdim),
            nn.SiLU(),
            nn.Linear(self.tdim, self.tdim))
        
        
        self.Con_embed = nn.Sequential(
                    nn.Linear(self.cdim, self.tdim),
                    nn.SiLU(),
                    nn.Linear(self.tdim, self.tdim))
        
        '''
        
        self.time_embed = nn.Sequential(
            nn.Linear(self.tdim, self.tdim),
            nn.SiLU(),
            nn.Linear(self.tdim, self.tdim))
       
        
    def LinLayer(self, dimi, dimo):
        
        return nn.Sequential(nn.Linear(dimi,dimo),
                             nn.SiLU(),
                             nn.BatchNorm1d(dimo),
                             nn.Dropout(p=0.1))
        


    def forward(self, x, timesteps):
        
                
        x = self.X_embed(x) + self.time_embed(timestep_embedding(timesteps, self.tdim))
        res_x = x
        
        for i in range(0,len(self.fc)):
            x = self.fc[i](x)
    
        x = torch.add(x,res_x)
        
        x = self.finalLayer(x)
        
        return x

# First Step: make a classifier object:
class Classifier_Model(torch.nn.Module):
    def __init__(self, Dict):
        nn.Module.__init__(self)
        
        self.xdim = Dict['xdim']
        self.tdim = Dict['tdim']
        self.cdim = Dict['cdim']

        self.net = Dict['net']
        
        self.fc = nn.ModuleList()
        
        
        self.time_embed = nn.Sequential(
            nn.Linear(self.tdim, self.tdim),
            nn.SiLU(),
            nn.Linear(self.tdim, self.tdim))
        
        self.X_embed = nn.Linear(self.xdim, self.tdim)
        
        self.fc.append(self.LinLayer(self.tdim,self.net[0]))
        '''
        self.fc.append(self.LinLayer(self.xdim,self.net[0]))
        '''
        
        for i in range(1, len(self.net)):
            self.fc.append(self.LinLayer(self.net[i-1],self.net[i]))
            
        
        self.fc.append(nn.Sequential(nn.Linear(self.net[-1], self.cdim), nn.Sigmoid()))

    def LinLayer(self, dimi, dimo):
        
        return nn.Sequential(nn.Linear(dimi,dimo),
                             nn.SiLU(),
                             #nn.BatchNorm1d(dimo),
                             nn.Dropout(p=0.1))
        

    def forward(self, x):
        
        x = self.X_embed(x)
        
        for i in range(0,len(self.fc)):
            x = self.fc[i](x)
    
        
        return x
    
    
# Make a regression Model and define training loop:
class Regression_ResNet_Model(torch.nn.Module):
    def __init__(self, Reg_Dict):
        nn.Module.__init__(self)
        
        self.xdim = Reg_Dict['xdim']
        self.ydim = Reg_Dict['ydim']
        self.tdim = Reg_Dict['tdim']
        self.net = Reg_Dict['net']
        
        self.fc = nn.ModuleList()
        
        self.fc.append(self.LinLayer(self.tdim,self.net[0]))
        
        for i in range(1, len(self.net)):
            self.fc.append(self.LinLayer(self.net[i-1],self.net[i]))
            
        self.fc.append(self.LinLayer(self.net[-1], self.tdim))
        
        
        self.finalLayer = nn.Sequential(nn.Linear(self.tdim, self.ydim))
        
    
        self.X_embed = nn.Linear(self.xdim, self.tdim)
       
        
    def LinLayer(self, dimi, dimo):
        
        return nn.Sequential(nn.Linear(dimi,dimo),
                             nn.SiLU(),
                             nn.LayerNorm(dimo),
                             nn.Dropout(p=0.2))
        


    def forward(self, x):

                
        x = self.X_embed(x) 
        res_x = x
        
        for i in range(0,len(self.fc)):
            x = self.fc[i](x)
    
        x = torch.add(x,res_x)
        
        x = self.finalLayer(x)
        
        return x

    
'''
==============================================================================
EMA - Exponential Moving Average: Helps with stable training
========================================================================
EMA class from: https://github.com/azad-academy/denoising-diffusion-model/blob/main/ema.py

'''
# Exponential Moving Average Class
# Orignal source: https://github.com/acids-ircam/diffusion_models


class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
        
        
'''
==========================================
Set up the data normalizer class
==========================================

'''        

class Data_Normalizer:
    def __init__(self, X_LL_Scaled, X_UL_Scaled,datalength):
        
        self.normalizer = PP.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(datalength // 30, 1000), 10),
            subsample=int(1e9)
            )
        
        self.X_LL_Scaled = X_LL_Scaled
        self.X_UL_Scaled = X_UL_Scaled
        
        self.X_LL_norm = np.zeros((1,len(X_LL_Scaled)))
        self.X_UL_norm = np.zeros((1,len(X_LL_Scaled)))
        
        self.X_mean = np.zeros((1,len(X_LL_Scaled)))
        self.X_std = np.zeros((1,len(X_LL_Scaled)))
        
    def fit_Data(self,X):
        
        
        
        x = 2.0*(X-self.X_LL_Scaled)/(self.X_UL_Scaled- self.X_LL_Scaled) - 1.0
        
        self.normalizer.fit(x)
        x = self.normalizer.transform(x) # Scale Dataset between 
        #x = (X-self.X_LL_Scaled)/(self.X_UL_Scaled- self.X_LL_Scaled)
        

        return x
    
    def transform_Data(self,X):
        x = 2.0*(X-self.X_LL_Scaled)/(self.X_UL_Scaled- self.X_LL_Scaled) - 1.0
        
        
        x = self.normalizer.transform(x)
        return x
        

    def scale_X(self,z):
        #rescales data
        z = self.normalizer.inverse_transform(z)
        scaled = (z + 1.0) * 0.5 * (self.X_UL_Scaled - self.X_LL_Scaled) + self.X_LL_Scaled
        #scaled = z* (self.X_UL_Scaled - self.X_LL_Scaled) + self.X_LL_Scaled

        '''
        x = self.normalizer.inverse_transform(x)
        
        #scaled = x* (self.X_UL_norm - self.X_LL_norm) + self.X_LL_norm
        '''
        #z = (z + 1.0) * 0.5 * (8.0) + 4.0
       
        #scaled = z*self.X_std + self.X_mean
        #scaled = self.normalizer.inverse_transform(scaled)
        return scaled     
        
        
        
        
'''
=======================================================================
Trainer class modified from Tab-ddpm paper code with help from hugging face
=====================================================================
'''
class GuidedDiffusionEnv:
    def __init__(self, DDPM_Dict, Class_Dict, Reg_Dict, X,Y, Cons, X_neg, Cons_neg):
        
        self.DDPM_Dict = DDPM_Dict
        self.Class_Dict = Class_Dict
        self.Reg_Dict = Reg_Dict
        
        self.device =torch.device(self.DDPM_Dict['device_name'])
        
        #Build the Diffusion Network
        self.diffusion = Denoise_ResNet_Model(self.DDPM_Dict)
        #Build Classifier Network
        self.classifier = Classifier_Model(self.Class_Dict)
        #Build Regression Networks:
        self.regressors = [Regression_ResNet_Model(self.Reg_Dict) for i in range(0,self.Reg_Dict['num_regressors'])]
        self.num_regressors = self.Reg_Dict['num_regressors']
        #self.load_trained_regressors()
        
        self.diffusion.to(self.device)
        self.classifier.to(self.device)

        for i in range(0,self.num_regressors):
            self.regressors[i].to(self.device)

        self.dataLength = self.DDPM_Dict['datalength']
        self.batch_size = self.DDPM_Dict['batch_size']
        self.gamma = self.DDPM_Dict['gamma']
        self.lambdas = np.array(self.DDPM_Dict['lambdas'])
        
        self.data_norm = Data_Normalizer(np.array(self.DDPM_Dict['X_LL']),np.array(self.DDPM_Dict['X_UL']),self.dataLength)
    
        
        
        self.X = self.data_norm.fit_Data(X)
        
        self.X_neg = self.data_norm.transform_Data(X_neg)
        
        #X and Y are numpy arrays - convert to tensor
        self.X = torch.from_numpy(self.X.astype('float32'))
        
        self.X_neg = torch.from_numpy(self.X_neg.astype('float32'))
        self.Y = torch.from_numpy(Y.astype('float32'))
        
        self.Cons = torch.from_numpy(Cons.astype('float32'))
        
        self.Cons_neg = torch.from_numpy(Cons_neg.astype('float32'))

    
        self.X = self.X.to(self.device)
        self.X_neg = self.X_neg.to(self.device)
        self.Y = self.Y.to(self.device)
        self.Cons = self.Cons.to(self.device)
        self.Cons_neg = self.Cons_neg.to(self.device)
        
        self.eps = 1e-8
        
        self.ema = EMA(0.99)
        self.ema.register(self.diffusion)
        
        
        #set up optimizer 
        self.timesteps = self.DDPM_Dict['Diffusion_Timesteps']
        self.num_diffusion_epochs = self.DDPM_Dict['Training_Epochs']
        
        self.num_classifier_epochs = self.Class_Dict['Training_Epochs']
        self.num_regressor_epochs = self.Reg_Dict['Training_Epochs']
        
        lr = self.DDPM_Dict['lr']
        self.init_lr = lr
        weight_decay = self.DDPM_Dict['weight_decay']
        
        self.optimizer_diffusion = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.optimizer_classifier = torch.optim.AdamW(self.classifier.parameters(),lr=.001, weight_decay=weight_decay)
        self.optimizer_regressors = [torch.optim.AdamW(self.regressors[i].parameters(),lr=.001, weight_decay=weight_decay) for i in range(0,self.Reg_Dict['num_regressors'])]
        


        self.log_every = 100
        self.print_every = 5000
        self.loss_history = []
        
        
        
        #Set up alpha terms
        self.betas = torch.linspace(0.001, 0.2, self.timesteps).to(self.device)
        
        #self.betas = betas_for_alpha_bar(self.timesteps, lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,)
        #self.betas = torch.from_numpy(self.betas.astype('float32')).to(self.device)
        
        self.alphas = 1. - self.betas
        
        self.log_alpha = torch.log(self.alphas)
        self.log_cumprod_alpha = np.cumsum(self.log_alpha.cpu().numpy())
        
        self.log_cumprod_alpha = torch.tensor(self.log_cumprod_alpha,device=self.device)
        
        
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1],[1,0],'constant', 0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod =  torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        a = torch.clone(self.posterior_variance)
        a[0] = a[1]                 
        self.posterior_log_variance_clipped = torch.log(a)
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev)* torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))

    """++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Start the training model functions
    """
    def extract(self,a, t, x_shape):
        b, *_ = t.shape
        t = t.to(a.device)
        out = a.gather(-1, t)
        while len(out.shape) < len(x_shape):
            out = out[..., None]
        return out.expand(x_shape)
    
    def _anneal_lr(self, epoch_step):
        #Update the learning rate
        frac_done = epoch_step / self.num_diffusion_epochs
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer_diffusion.param_groups:
            param_group["lr"] = lr
               
            
    '''
    =========================================================================
    Vanilla Diffusion
    ==========================================================================
    '''
    def q_sample(self,x_start, t, noise=None):
        """
        qsample from https://huggingface.co/blog/annotated-diffusion
        """
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)
    
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
    
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    
    def p_loss(self,x_start,t, noise=None,loss_type='l2'):
        '''
        from https://huggingface.co/blog/annotated-diffusion
        '''
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.diffusion(x_noisy, t)
        
        #predicted_noise = predicted_noise.clamp(-3,3)
        
        if loss_type == 'l1':
            loss1 = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss1 = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss1 = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
                
        return loss1 
        
    '''
    ==============================================================================
    Classifier and Regression Training Functions
    ==============================================================================
    '''
           
    
    def run_classifier_step(self,x,cons):
        
        self.optimizer_classifier.zero_grad()
        

        predicted_cons = self.classifier(x)
        
        loss = F.binary_cross_entropy(predicted_cons, cons) #F.mse_loss(predicted_cons, cons) #F.binary_cross_entropy(predicted_cons, cons)
        loss.backward()
        self.optimizer_classifier.step()
        
        return loss
    
    def run_train_classifier_loop(self, batches_per_epoch=100):
        
        X = torch.cat((self.X,self.X_neg))
        C = torch.cat((self.Cons,self.Cons_neg))
        
        #print(C.shape)
        
        datalength = X.shape[0]
        
        print('Classifier Model Training...')
        self.classifier.train()
        
        num_batches = datalength // self.batch_size
        
        batches_per_epoch = min(num_batches,batches_per_epoch)
        
    
        x_batch = torch.full((batches_per_epoch,self.batch_size,self.classifier.xdim), 0, dtype=torch.float32,device=self.device)
        
        cons_batch = torch.full((batches_per_epoch,self.batch_size,self.classifier.cdim), 0, dtype=torch.float32,device=self.device)
        
        for i in tqdm(range(self.num_classifier_epochs)):
            
            #IDX = permute_idx(self.dataLength) # get randomized list of idx for batching
            
            for j in range(0,batches_per_epoch):
                
                A = np.random.randint(0,datalength,self.batch_size)
                x_batch[j] = X[A] 
                #y_batch[j] = self.Y[IDX[j*self.batch_size:(j+1)*self.batch_size]] 
                cons_batch[j] = C[A]
                #cons_batch[j] = self.Cons[IDX[j*self.batch_size:(j+1)*self.batch_size]]
            
            for j in range(0,batches_per_epoch):
                       
                loss = self.run_classifier_step(x_batch[j],cons_batch[j])    
        '''    
        for i in tqdm(range(0,self.num_classifier_epochs)):
            loss = self.run_classifier_step(X,C)  
        '''   
        self.classifier.eval()
        
        A = np.random.randint(0,datalength,1000)
        
        
        C_pred = self.classifier(X[A])
        

        C_pred = C_pred.to(torch.device('cpu')).detach().numpy()
       
        #print(C_pred.shape)
        C_pred = np.rint(C_pred) #Make it an iteger guess

        C = C.to(torch.device('cpu')).detach().numpy()
      
        F1 = f1_score(C[A],C_pred)

        print('F1 score: ' + str(F1))
        
        print('Classifier Training Complete!')



    def run_regressor_step(self,x,y,idx):
        self.optimizer_regressors[idx].zero_grad()
        

        predicted_y = self.regressors[idx](x)
        
        loss =  F.mse_loss(predicted_y, y) 
        loss.backward()
        self.optimizer_regressors[idx].step()
        
        return loss  
    
    def run_train_regressors_loop(self,batches_per_epoch=100):
            
            X = self.X
            Y = self.Y
            
            datalength = X.shape[0]
            
            num_batches = datalength // self.batch_size
        
            batches_per_epoch = min(num_batches,batches_per_epoch)


            
            print('Regressor Model Training...')
            for k in range(0,self.num_regressors):
                print('Training Regression for Objective: ' + str(k))
                self.regressors[k].train()

                x_batch = torch.full((batches_per_epoch,self.batch_size,self.classifier.xdim), 0, dtype=torch.float32,device=self.device)
        
                y_batch = torch.full((batches_per_epoch,self.batch_size,self.regressors[k].ydim), 0, dtype=torch.float32,device=self.device)
        


                for i in tqdm(range(0,self.num_regressor_epochs)):

                    for j in range(0,batches_per_epoch):
                        
                        A = np.random.randint(0,datalength,self.batch_size)
                        x_batch[j] = X[A] 
                        y_batch[j] = Y[A,k:k+1] 
                    
                    for j in range(0,batches_per_epoch):

                        loss = self.run_regressor_step(x_batch[j],y_batch[j],k)   
                     
            
                print('Regression Model Training for Objective ' + str(k) + ' Complete!')

                self.regressors[k].eval()
    
                Y_pred = self.regressors[k](X)


                Y_pred = Y_pred.to(torch.device('cpu')).detach().numpy()
                y = Y[:,k].to(torch.device('cpu')).detach().numpy()

                Rsq = r2_score(y, Y_pred)
                print("R2 score of Y:" + str(Rsq))

            print('Regressor Training Complete!')

    




    '''
    ==============================================================================
    Diffusion Training and Sampling Functions
    ==============================================================================
    '''      


    def run_diffusion_step(self, x):
        self.optimizer_diffusion.zero_grad()
        
        t = torch.randint(0,self.timesteps,(self.batch_size,),device=self.device)
        loss1 = self.p_loss(x,t,loss_type='l2')
        
        loss = loss1 
        loss.backward()
        self.optimizer_diffusion.step()

        return loss

    def run_train_diffusion_loop(self, batches_per_epoch=100):
        print('Denoising Model Training...')
        self.diffusion.train()
        
        num_batches = self.dataLength // self.batch_size
        
        batches_per_epoch = min(num_batches,batches_per_epoch)
        
    
        x_batch = torch.full((batches_per_epoch,self.batch_size,self.diffusion.xdim), 0, dtype=torch.float32,device=self.device)
        #y_batch = torch.full((batches_per_epoch,self.batch_size,self.diffusion.ydim), 0, dtype=torch.float32,device=self.device)
        #cons_batch = torch.full((batches_per_epoch,self.batch_size,self.diffusion.cdim), 0, dtype=torch.float32,device=self.device)
        
        for i in tqdm(range(self.num_diffusion_epochs)):
            
            #IDX = permute_idx(self.dataLength) # get randomized list of idx for batching
            
            for j in range(0,batches_per_epoch):
                
                A = np.random.randint(0,self.dataLength,self.batch_size)
                x_batch[j] = self.X[A] 
                #y_batch[j] = self.Y[IDX[j*self.batch_size:(j+1)*self.batch_size]] 
                #cons_batch[j] = self.Cons[A]
                #cons_batch[j] = self.Cons[IDX[j*self.batch_size:(j+1)*self.batch_size]]
            
            for j in range(0,batches_per_epoch):
                       
                loss = self.run_diffusion_step(x_batch[j])
                '''
                Gaussian Diffusion (oooohhhh ahhhhhh) from TabDDPM:
                '''
                #loss = self.train_step(x_batch[j])

            self._anneal_lr(i)

            if (i + 1) % self.log_every == 0:
                self.loss_history.append([i+1,float(loss.to('cpu').detach().numpy())])
        
            if (i + 1) % self.print_every == 0:
                    print(f'Step {(i + 1)}/{self.num_diffusion_epochs} Loss: {loss}')
                

            self.ema.update(self.diffusion)
        #Make Loss History an np array
        self.loss_history = np.array(self.loss_history)
        print('Denoising Model Training Complete!')
    
    def cond_fn(self, x, t, cons):
        #From OpenAI: https://github.com/openai/guided-diffusion/blob/main/scripts/classifier_sample.py
        
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            
            pred_cons = self.classifier(x_in)
            
            error = (cons-pred_cons)**2.0 #F.binary_cross_entropy(pred_cons, cons) #
            
            #log_p = torch.log(pred_cons)
            
            #sign = torch.sign(cons-0.5)

            grad = torch.autograd.grad(error.sum(), x_in)[0] 
            
            #print(grad[0])          
            return -grad
        
    def perf_fn(self, x, idx):
        #From OpenAI: https://github.com/openai/guided-diffusion/blob/main/scripts/classifier_sample.py
        
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            
            perf = self.regressors[idx](x_in)
            

            grad = torch.autograd.grad(perf.sum(), x_in)[0] 
            
            #print(grad[0])          
            return grad
        
    @torch.no_grad()
    def p_sample(self, x, t, cons):
        
        time= torch.full((x.size(dim=0),),t,dtype=torch.int64,device=self.device)
        
        X_diff = self.diffusion(x, time) 
        
        
        betas_t = self.extract(self.betas, time, x.shape)
        
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, time, x.shape
        )
        
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, time, x.shape)
        
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
    
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * X_diff/ sqrt_one_minus_alphas_cumprod_t
        )
        
        
        posterior_variance_t = self.extract(self.posterior_variance, time, x.shape)
        
        
        cons_grad = self.cond_fn(x, time, cons)
        #print(gradient.detach().to('cpu')[0])

        if t == 0:
            return model_mean
        else:
            
            noise = torch.randn_like(x,device=self.device)
            # Dot product gradient to noise
            return model_mean + torch.sqrt(posterior_variance_t) * (noise*(1.0-self.gamma) + self.gamma*cons_grad.float())
       
        
    @torch.no_grad()
    def Performance_p_sample(self, x, t, cons,perf_weights):
        
         
        time= torch.full((x.size(dim=0),),t,dtype=torch.int64,device=self.device)
        
        X_diff = self.diffusion(x, time) 
        
        
        betas_t = self.extract(self.betas, time, x.shape)
        
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, time, x.shape
        )
        
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, time, x.shape)
        
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
    
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * X_diff/ sqrt_one_minus_alphas_cumprod_t
        )
        
        
        posterior_variance_t = self.extract(self.posterior_variance, time, x.shape)
        
        
        #cons_gradient = self.cond_fn(x, time, cons)
        #print(gradient.detach().to('cpu')[0])

        if t == 0:
            return model_mean
        else:
            
            noise = torch.randn_like(x,device=self.device)
            # Dot product gradient to noise
            perf_guidance = torch.zeros_like(x,dtype=torch.float32,device=self.device)
            
            
            for i in range(0,len(self.regressors)):                
                perf_guidance = perf_guidance + self.perf_fn(model_mean,i)*perf_weights[i]
                
            
            #perf_grad = self.perf_fn(model_mean,0) 
            cons_grad = self.cond_fn(model_mean, time, cons)
              
            return model_mean + torch.sqrt(posterior_variance_t) * (noise*(1.0-self.gamma) + self.gamma*cons_grad.float() - perf_guidance)
            
            #return model_mean - self.lam* perf_grad.float()
        
    @torch.no_grad()
    def gen_samples(self, cons):
        #COND is a numpy array of the conditioning it is shape (num_samples,conditioning terms)
        num_samples = len(cons)

        cons = torch.from_numpy(cons.astype('float32'))
        cons = cons.to(self.device)
        
        #print(num_samples) #should be 1
        
        x_gen = torch.randn((num_samples,self.diffusion.xdim),device=self.device)
        
        self.diffusion.eval()
        self.classifier.eval()
        
   
        for i in tqdm(range(self.timesteps - 1, 0, -1)):


            x_gen = self.p_sample(x_gen, i,cons)

        
        output = x_gen.cpu().detach().numpy()
            
            
        output_scaled = self.data_norm.scale_X(output)
        
        return output_scaled, output
    
    
    @torch.no_grad()
    def gen_perf_samples(self, cons,weights):
        #COND is a numpy array of the conditioning it is shape (num_samples,conditioning terms)
        num_samples = len(cons)
        
        perf_time_ratio = 1.0 -0.8

        cons = torch.from_numpy(cons.astype('float32'))
        cons = cons.to(self.device)
        
        #print(num_samples) #should be 1
        
        x_gen = torch.randn((num_samples,self.diffusion.xdim),device=self.device)
        
        perf_weights = torch.zeros((len(self.lambdas),num_samples,self.diffusion.xdim),device=self.device)
        
        
        
        self.diffusion.eval()
        self.classifier.eval()
        
        for i in range(0,len(self.regressors)):
            self.regressors[i].eval()
            A = self.lambdas[i]*weights[:,i]
            A = A.reshape((len(A),1))
            perf_weights[i,:,:] = torch.from_numpy(A.astype('float32')).to(self.device).repeat(1,self.diffusion.xdim)

        

        #print(perf_weights.shape)      

        for i in tqdm(range(self.timesteps - 1, int(perf_time_ratio*self.timesteps), -1)):

            x_gen = self.Performance_p_sample(x_gen, i,cons,perf_weights)
            
            
               
        for i in tqdm(range(int(perf_time_ratio*self.timesteps), 0, -1)):


            x_gen = self.p_sample(x_gen, i,cons)

        
        output = x_gen.cpu().detach().numpy()
            
            
        output_scaled = self.data_norm.scale_X(output)
        
        return output_scaled, output
          
    def Predict_Perf_numpy(self,X):
        X_norm = self.data_norm.transform_Data(X)
        
        X_norm = torch.from_numpy(X_norm.astype('float32')).to(self.device)
        
        Y_pred = torch.full((len(X),len(self.regressors)),0.0,dtype=torch.float32,device=self.device)

        for i in range(0,len(self.regressors)):
            Y_pred[:,i:i+1] = self.regressors[i](X_norm)

        Y_pred = Y_pred.to('cpu').detach().numpy()
        
        return Y_pred
    
    def Predict_Perf_Tensor(self,X_norm):

        Y_pred = torch.full((len(X_norm),len(self.regressors)),0.0,dtype=torch.float32,device=self.device)

        for i in range(0,len(self.regressors)):
            Y_pred[:,i:i+1] = self.regressors[i](X_norm)

        
        return Y_pred
    
    '''
    ==============================================================================
    Saving and Loading Model Functions
    ==============================================================================
    '''

    def load_trained_diffusion_model(self,PATH):
        #PATH is full path to the state dictionary, including the file name and extension
        self.diffusion.load_state_dict(torch.load(PATH))
    
    def Load_Dict(PATH):
        #returns the dictionary for the DDPM_Dictionary to rebuild the model
        #PATH is the path including file name and extension of the json file that stores it. 
        f = open(PATH)
        return json.loads(f)
        
    
    def Save_diffusion_model(self,PATH,name):
        '''
        PATH is the path to the folder to store this in, including '/' at the end
        name is the name of the model to save without an extension
        '''
        torch.save(self.diffusion.state_dict(), PATH+name+'_diffusion.pth')
        
        JSON = json.dumps(self.DDPM_Dict)
        f = open(PATH+name+'.json', 'w')
        f.write(JSON)
        f.close()
        
    def load_trained_classifier_model(self,PATH):
        #PATH is full path to the state dictionary, including the file name and extension
        self.classifier.load_state_dict(torch.load(PATH))
        
    def load_trained_regressors(self):
        labels = self.Reg_Dict['Model_Paths']
        
        for i in range(0,len(labels)):
            self.regressors.append(Regression_ResNet_Model(self.Reg_Dict))
            self.regressors[i].load_state_dict(torch.load(labels[i]))
            self.regressors[i].to(self.device)
        
    
    def Save_classifier_model(self,PATH,name):
        '''
        PATH is the path to the folder to store this in, including '/' at the end
        name is the name of the model to save without an extension
        '''
        torch.save(self.classifier.state_dict(), PATH+name+ '.pth')
        
        JSON = json.dumps(self.Class_Dict)
        f = open(PATH+name+ '.json', 'w')
        f.write(JSON)
        f.close()
    
    def Save_regression_models(self,PATH):
        '''
        PATH is the path to the folder to store this in, including '/' at the end
       
        '''
        for i in range(0,len(self.regressors)):
            torch.save(self.regressors[i].state_dict(), PATH + self.Reg_Dict['Model_Labels'][i] +'.pth')
        
        JSON = json.dumps(self.Reg_Dict)
        f = open(PATH + '_regressor_Dict.json', 'w')
        f.write(JSON)
        f.close()




