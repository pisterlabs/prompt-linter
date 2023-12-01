#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:30:04 2018

@author: subhayanmukherjee
"""

import glob
from data_utils import saturate_outlier, imshow
from keras.layers import Input, Dense, Conv2D, SeparableConv2D, MaxPooling2D, UpSampling2D, Concatenate, Lambda
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.transform import resize
from keras.models import load_model
import h5py
from coherence import estimate_coherence
from skimage.segmentation import chan_vese
from skimage.segmentation import felzenszwalb
from skimage.measure import label
from skimage.util import pad
from scipy import ndimage, stats
from time import time
import math

ifg_hdf5_path = '/home/subhayanmukherjee/Documents/MRC-3vG-CARIC/suba/cohqal/testieees/train/ifg_patches.hdf5'
coh_hdf5_path = '/home/subhayanmukherjee/Documents/MRC-3vG-CARIC/suba/cohqal/testieees/train/coh_patches.hdf5'


train_path = '/home/subhayanmukherjee/Documents/MRC-3vG-CARIC/suba/cohqal/testieees/simtdset/'
train_filelist = glob.glob(train_path + '*.npy')

epoch_path = '/home/subhayanmukherjee/Documents/MRC-3vG-CARIC/suba/cohqal/testieees/train/'
ifg_ae_weight_path = epoch_path + 'ifg_ae/weights.{epoch:02d}.hdf5'
coh_nw_weight_path = epoch_path + 'coh_nw/weights.{epoch:02d}.hdf5'


def readFloatComplex(fileName, width=1):
    return np.fromfile(fileName,'>c8').astype(np.complex).reshape(-1, width)

def process_ifg(input_ifg):
    Z_processed = saturate_outlier(input_ifg)
    Z_real = Z_processed.real + 1
    Z_real = np.expand_dims(Z_real, axis=-1)
    Z_imag = Z_processed.imag + 1
    Z_imag = np.expand_dims(Z_imag, axis=-1)
    return np.concatenate((Z_real, Z_imag), axis=-1)

def build_ifg(input_pred):
    out_ifg_real = input_pred[:,:,:,0]
    out_ifg_imag = input_pred[:,:,:,1]
    out_ifg = out_ifg_real + out_ifg_imag * 1j
    out_ifg -= (1 + 1j)
    return out_ifg

def resize_pred(pred,out_height,out_width):
    out_res = np.zeros((1,out_height,out_width,2),dtype=np.float)
    
    pred[0,:,:,0] = np.clip(pred[0,:,:,0] - 1, a_min=-1.0, a_max=1.0)
    pred[0,:,:,1] = np.clip(pred[0,:,:,1] - 1, a_min=-1.0, a_max=1.0)
    
    out_res[0,:,:,0] = resize(pred[0,:,:,0], (out_height,out_width))
    out_res[0,:,:,1] = resize(pred[0,:,:,1], (out_height,out_width))
    
    out_res[0,:,:,0] = out_res[0,:,:,0] + 1
    out_res[0,:,:,1] = out_res[0,:,:,1] + 1
    
    return out_res

def generate_ifg_dataset(source_files, hdf5_file, pat_per_ifg):
    examples_cnt = len(source_files)
    
    rand_idx = np.random.permutation(examples_cnt)
    for loop_idx in range(examples_cnt):
        path_in_str = source_files[rand_idx[loop_idx]]
        
        train_image = np.load(path_in_str)
        train_image = np.angle(train_image)
        train_image = np.cos(train_image) + 1j*np.sin(train_image)
        
        Z_ab = process_ifg(train_image)
        train_pat = extract_patches_2d( Z_ab, (60,60), max_patches=pat_per_ifg, random_state=0 )
        
        hdf5_file["train_img"][loop_idx*pat_per_ifg : (loop_idx+1)*pat_per_ifg, ...] = train_pat
        hdf5_file["train_lab"][loop_idx*pat_per_ifg : (loop_idx+1)*pat_per_ifg, ...] = train_pat

def generate_coh_dataset(source_files, hdf5_file, pat_per_ifg, ifg_ae):
    examples_cnt = len(source_files)
    
    rand_idx = np.random.permutation(examples_cnt)
    for loop_idx in range(examples_cnt):
        path_in_str = source_files[rand_idx[loop_idx]]
        
        train_image = np.load(path_in_str)
        train_image = np.angle(train_image)
        train_image = np.cos(train_image) + 1j*np.sin(train_image)
        
        Z_ab = process_ifg(train_image)
        train_rec = np.squeeze(build_ifg(resize_pred( ifg_ae.predict(np.expand_dims( Z_ab, axis=0 )) ,1000,1000 )))
        
        padded_train_image = pad(train_image, (5,5), 'edge')
        padded_train_rec = pad(train_rec, (5,5), 'edge')
        
        pd_img, org_cropped = estimate_coherence(padded_train_image, padded_train_rec, 11)
        pd_abs = np.absolute(pd_img)
        pd_seg = chan_vese(pd_abs, mu=0.01)    # , tol=0.0001)
        
        if np.mean(pd_abs[pd_seg==True]) < np.mean(pd_abs[pd_seg==False]):
            bg_int = 0
        else:
            bg_int = 1
        
        pd_lab = label(pd_seg.astype(np.uint8), background=bg_int)
        
        sharp_coh = np.ones((1000,1000),dtype=np.float32)
        pd_abs[pd_lab==0] = sharp_coh[pd_lab==0]
        
        mx_lab = np.amax(pd_lab)
        for lab in range(mx_lab + 1):
            se_lab = lab + 1
            se_pix = ( pd_lab==se_lab )
            se_coh = pd_abs[se_pix]
            se_avg = np.mean( se_coh )
            se_std = np.std( se_coh )
            pd_abs[se_pix] = se_avg - se_std
        
        Z_ab = process_ifg(org_cropped)
        
        train_pat = extract_patches_2d(   Z_ab, (64,64), max_patches=pat_per_ifg, random_state=0 )
        train_lab = extract_patches_2d( pd_abs, (64,64), max_patches=pat_per_ifg, random_state=0 )
        
        hdf5_file["train_img"][loop_idx*pat_per_ifg : (loop_idx+1)*pat_per_ifg, ...] = train_pat
        hdf5_file["train_lab"][loop_idx*pat_per_ifg : (loop_idx+1)*pat_per_ifg, ...] = np.expand_dims( train_lab, axis=-1 )


def generate_data(hdf5_file, batch_size, examples_cnt):
    batch_cnt = examples_cnt/batch_size
    while 1:
        rand_idx = np.random.permutation(int(batch_cnt))*batch_size
        loop_idx = 0
        while loop_idx < batch_cnt:
            data = hdf5_file["train_img"][rand_idx[loop_idx]:(rand_idx[loop_idx]+batch_size)]
            labels = hdf5_file["train_lab"][rand_idx[loop_idx]:(rand_idx[loop_idx]+batch_size)]
            
            yield (data, labels)
            loop_idx += 1


def suba_reg(weight_matrix):
    return K.std(weight_matrix)


def create_ifg_ae():
    input_img = Input(shape=(None, None, 2))
        
    x1  = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x2  = Conv2D(8, (3, 3), activation='relu', padding='same')(x1)
    x3  = MaxPooling2D((3, 3), padding='same')(x2)
    x4  = Conv2D(8, (3, 3), activation='relu', padding='same')(x3)
    x5  = UpSampling2D((3, 3))(x4)
    x6  = Conv2D(16, (3, 3), activation='relu', padding='same')(x5)
    decoded = Conv2D(2, (3, 3), activation='relu', padding='same')(x6)
    
    ifg_ae = Model(input_img, decoded)
    ifg_ae.compile(optimizer='adam', loss='mean_squared_error')
    
    return ifg_ae

def create_coh_nw():
    input_img = Input(shape=(None, None, 2))
    
    x1  = Conv2D(4, (3, 3), activation='relu', padding='same')(input_img)
    x1  = Conv2D(4, (3, 3), activation='relu', padding='same')(x1)
    x1  = Conv2D(4, (3, 3), activation='relu', padding='same', kernel_regularizer=suba_reg)(x1)
    
    output_img = Conv2D( 1, (3, 3), activation='sigmoid', padding='same')(x1)
    
    coh_nw = Model(input_img, output_img)
    coh_nw.compile(optimizer='adam', loss='mean_squared_error')
    
    return coh_nw


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


create_ifg = True
#var = input("Create ifg dataset ? (y/n) : ")
#if var == 'y':
#    create_ifg = True
#else:
#    create_ifg = False


pat_per_ifg  = 500
patch_height = 60
patch_width  = 60
batch_size   = 8
epochs       = 50


if create_ifg:
    hdf5_file = h5py.File(ifg_hdf5_path, mode='w')
    dataset_length = len(train_filelist)
    train_shape = label_shape = (dataset_length*pat_per_ifg, patch_height, patch_width, 2)
    hdf5_file.create_dataset("train_img", train_shape, np.float32)
    hdf5_file.create_dataset("train_lab", label_shape, np.float32)
    
    generate_ifg_dataset(train_filelist, hdf5_file, pat_per_ifg)    
    hdf5_file.close()


hdf5_file = h5py.File(ifg_hdf5_path, mode='r')
train_num, patch_height, patch_width, num_recons = hdf5_file["train_img"].shape

ifg_ae_cpk = ModelCheckpoint(ifg_ae_weight_path, period=5)
ifg_ae = create_ifg_ae()
ifg_ae.fit_generator(initial_epoch=0, epochs=epochs, generator=generate_data(hdf5_file,batch_size,train_num), steps_per_epoch=train_num/batch_size, callbacks=[ifg_ae_cpk])
hdf5_file.close()


create_coh = True
#var = input("Create coh dataset ? (y/n) : ")
#if var == 'y':
#    create_coh = True
#else:
#    create_coh = False


patch_height = 64
patch_width = 64


if create_coh:
    hdf5_file = h5py.File(coh_hdf5_path, mode='w')
    dataset_length = len(train_filelist)
    train_shape = (dataset_length*pat_per_ifg, patch_height, patch_width, 2)
    label_shape = (dataset_length*pat_per_ifg, patch_height, patch_width, 1)
    hdf5_file.create_dataset("train_img", train_shape, np.float32)
    hdf5_file.create_dataset("train_lab", label_shape, np.float32)
    
    generate_coh_dataset(train_filelist, hdf5_file, pat_per_ifg, ifg_ae)    
    hdf5_file.close()


hdf5_file = h5py.File(coh_hdf5_path, mode='r')
train_num, patch_height, patch_width, num_recons = hdf5_file["train_img"].shape

coh_nw_cpk = ModelCheckpoint(coh_nw_weight_path, period=5)
coh_nw = create_coh_nw()
coh_nw.fit_generator(initial_epoch=0, epochs=epochs, generator=generate_data(hdf5_file,batch_size,train_num), steps_per_epoch=train_num/batch_size, callbacks=[coh_nw_cpk])
hdf5_file.close()
