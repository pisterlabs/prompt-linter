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
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.transform import resize
from keras.models import load_model
import h5py
from coherence import estimate_coherence
from skimage.util import pad
from time import time
import math

import sys
sys.path.append("./pygco")
import pygco

hdf5_path = '/home/subhayanmukherjee/Documents/MRC-3vG-CARIC/suba/together_cnn/train/coh_ae/coh_patches.hdf5'


train_path = '/path/to/training/dataset/interferograms/'
train_filelist = glob.glob(train_path + '*.interferogram.file.extension')

test_path = '/path/to/testing/dataset/interferograms/'
test_filelist = glob.glob(test_path + '*.interferogram.file.extension')

epoch_path = '/home/subhayanmukherjee/Documents/MRC-3vG-CARIC/suba/together_cnn/train/coh_ae/'
weight_path = epoch_path + 'weights.{epoch:02d}.hdf5'

pat_per_ifg = 500
patch_height = 64
patch_width = 64
batch_size = 100


var = input("Create coherence ? (y/n) : ")

if var == 'y':
    create_coh = True
else:
    create_coh = False

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


def generate_training_labels(source_files, hdf5_file, pat_per_ifg, ifg_ae):
    examples_cnt = len(source_files)
    
    rand_idx = np.random.permutation(examples_cnt)
    for loop_idx in range(examples_cnt):
        path_in_str = source_files[rand_idx[loop_idx]]
        
        train_image = readFloatComplex( path_in_str, 1000*1000  )
        train_image = np.reshape( train_image,      (1000,1000) )
        
        Z_ab = process_ifg(train_image)
        train_rec = np.squeeze(build_ifg(resize_pred( ifg_ae.predict(np.expand_dims( Z_ab, axis=0 )) ,1000,1000 )))
        
        padded_train_image = pad(train_image, (3,3), 'edge')
        padded_train_rec = pad(train_rec, (3,3), 'edge')
        
        pd_img, org_cropped = estimate_coherence(padded_train_image, padded_train_rec, 7)
        pd_abs = np.absolute(pd_img)
        plt.imsave('training_rawlb_' + str(loop_idx) + '.png', pd_abs, cmap='gray', vmin=0, vmax=1)
        
        th_min = 0.6
        
        img = np.random.randn(1000, 1000)
        img[pd_abs >=th_min] += 2
        img -= 1
        
        unary = np.c_[img.reshape(img.size, 1), -img.reshape(img.size, 1)].copy()
        
        smooth = 1 - np.eye(2)

        unary = np.tile(img[:,:,np.newaxis], [1, 1, 2])
        unary[:, :, 0] = img
        unary[:, :, 1] = -img

        unary_new = unary.reshape((1000, 1000, 2))

        assert np.abs(unary - unary_new).max() == 0.
        assert not (unary != unary_new).any()
        
        label_s = pygco.cut_grid_graph_simple(unary_new, smooth * 2.5, n_iter=-1).reshape(1000, 1000)
        pd_abs = label_s.astype(np.float)
        
        Z_ab = process_ifg(org_cropped)
        
        train_pat = extract_patches_2d(   Z_ab, (64,64), max_patches=pat_per_ifg, random_state=0 )
        train_lab = extract_patches_2d( pd_abs, (64,64), max_patches=pat_per_ifg, random_state=0 )
        
        plt.imsave('training_label_' + str(loop_idx) + '.png', pd_abs, cmap='gray', vmin=0, vmax=1)
        
        # save the patches (and labels) extracted from current training image
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


class cb_PredIfg(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 0:	# monitor training progress by running model on a sample test interferogram after every five epochs
            test_ifg = '/path/to/sample/test/interferogram/interferogram_file_name.interferogram_file_extension'
            test_image = readFloatComplex( test_ifg, 1000*1000 )
            test_image = np.reshape( test_image, (1000,1000) )
            
            Z_ab = process_ifg(test_image)
            C_ab = np.expand_dims(Z_ab, axis=0)
            pred_err = self.model.predict(C_ab)
            pred_err = np.squeeze( pred_err )
            
            plt.imsave( epoch_path+str(epoch)+'.png', pred_err, cmap='gray', vmin=0, vmax=1.0 )


def create_coh_nw():
    input_img = Input(shape=(None, None, 2))  # adapt this if using `channels_first` image data format
    
    x1  = SeparableConv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x1  = SeparableConv2D(16, (3, 3), activation='relu', padding='same')(x1)
    x1  = SeparableConv2D(16, (3, 3), activation='relu', padding='same')(x1)
    
    output_img = SeparableConv2D( 1, (3, 3), activation='sigmoid', padding='same')(x1)
    
    coh_nw = Model(input_img, output_img)
    coh_nw.compile(optimizer='adam', loss='binary_crossentropy')
    
    return coh_nw


def lr_func(epoch):
    init_lr = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = init_lr * math.pow(drop, math.floor((1 + epoch)/epochs_drop))
    return lrate


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


pred_coh = cb_PredIfg()
coh_ae_cpk = ModelCheckpoint(weight_path, period=5)
ifg_ae = load_model('/path/to/trained/interferogram_filtering_cnn/checkpoint_filename.hdf5')
# ifg_ae = load_model('/home/subhayanmukherjee/Documents/MRC-3vG-CARIC/suba/together_cnn/train/MaxPoolCnt_1/encoded-20/weights.50.hdf5')
lr_schd = LearningRateScheduler(lr_func)

if create_coh:
    hdf5_file = h5py.File(hdf5_path, mode='w')
    dataset_length = len(train_filelist)
    train_shape = (dataset_length*pat_per_ifg, patch_height, patch_width, 2)
    label_shape = (dataset_length*pat_per_ifg, patch_height, patch_width, 1)
    hdf5_file.create_dataset("train_img", train_shape, np.float32)
    hdf5_file.create_dataset("train_lab", label_shape, np.float32)
    
    generate_training_labels(train_filelist, hdf5_file, pat_per_ifg, ifg_ae)    
    hdf5_file.close()

run_inference = True

if run_inference:
    test_image = readFloatComplex( '/path/to/sample/test/interferogram/interferogram_file_name.interferogram_file_extension', 1000*1000 )
    test_image = np.reshape( test_image, (1000,1000) )
    
    Z_ab = process_ifg(test_image)
    test_rec = np.squeeze(build_ifg(resize_pred( ifg_ae.predict(np.expand_dims( Z_ab, axis=0 )) ,1000,1000 )))
    
    Z_ab = np.expand_dims(Z_ab, axis=0)
    trained_coh = create_coh_nw()
    trained_coh.load_weights('/path/to/trained/coherence_classification_cnn/checkpoint_filename.hdf5')
    
    start_time = time()    
    pred_err = trained_coh.predict(Z_ab)
    print('Elapsed time: ' + str( time()-start_time ))
    
    pred_err = np.squeeze( pred_err )
    plt.figure()
    plt.imshow( pred_err, cmap='gray', vmin=0, vmax=1.0 )
    plt.colorbar()
    plt.show()
    plt.imsave( 'pred_phase.png', np.angle(test_rec), cmap='jet', vmin=-np.pi, vmax=np.pi )
    plt.imsave(    'pred_coh.png', pred_err, cmap='gray', vmin=0, vmax=1.0 )
    plt.imsave( 'input_phase.png', np.angle(test_image), cmap='jet', vmin=-np.pi, vmax=np.pi)
    raise SystemExit(0)

coh_ae = create_coh_nw()

hdf5_file = h5py.File(hdf5_path, mode='r')
train_num, patch_height, patch_width, num_recons = hdf5_file["train_img"].shape

coh_ae.fit_generator(initial_epoch=0, epochs=500, generator=generate_data(hdf5_file,batch_size,train_num), steps_per_epoch=train_num/batch_size, callbacks=[coh_ae_cpk,pred_coh,lr_schd])
