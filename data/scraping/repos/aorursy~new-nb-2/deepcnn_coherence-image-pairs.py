import datetime

import pandas as pd

import numpy as np

from scipy.io import loadmat

from operator import itemgetter

import random

import os

import time

import glob

from matplotlib import pyplot as plt

from pylab import plot, show, subplot, specgram, imshow

from scipy.signal import lfilter, butter

from matplotlib.mlab import cohere_pairs, cohere, magnitude_spectrum#, specgram



random.seed(2075)

np.random.seed(2075)

FILTER_N = 4  # Order of the filters to use

Fs = 400

NFFT = 1024

N = 16; ch = 16

m = 80000 # This will give 200 x 200 spectrogram

    

def mat_to_dataframe(path):

    mat = loadmat(path)

    names = mat['dataStruct'].dtype.names

    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}

    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0])



def butter_lowpass(X, highcut, fs=Fs, order=FILTER_N):

    nyq = 0.5 * fs

    high = highcut / nyq

    b, a = butter(order, high, btype="lowpass")

    return lfilter(b, a, X, axis=0)



def butter_bandpass(X, lowcut, highcut, fs=Fs, order=FILTER_N):

    nyq = 0.5 * fs

    cutoff = [lowcut / nyq, highcut / nyq]

    b, a = butter(order, cutoff, btype="bandpass")

    return lfilter(b, a, X, axis=0)



def butter_highpass(X, highcut, fs=Fs, order=FILTER_N):

    nyq = 0.5 * fs

    high = highcut / nyq

    b, a = butter(order, high, btype="highpass")

    return lfilter(b, a, X, axis=0)



def coherence(df, ch_idx):

    #X = df.as_matrix()

    X = df

    X = butter_bandpass(X, 0.00167, 180)

    X = X - X.mean(axis=0) / X.std(axis=0)

    Cxy, phase, freqs = cohere_pairs(X[:,:], ch_idx,

                                     NFFT=256, Fs=400, noverlap=128)

    

    img_p = np.array([phase[ch_idx[i]] for i in range(len(ch_idx))])

    img_m = np.array([Cxy[ch_idx[i]] for i in range(len(ch_idx))])

    

    #img = np.sqrt(np.square(img_m) + np.square(img_p))

    img = img_m

    

    img *= 255 / img.max()

    

    return img



def plot_coherence(pairs, ch_idx):

    for i in range(len(pairs)):

        pair = pairs[i]

    

        X0 = mat_to_dataframe(pair[0]).as_matrix()

        X1 = mat_to_dataframe(pair[1]).as_matrix()



        IMG0 = coherence(X0, ch_idx)

        IMG1 = coherence(X1, ch_idx)

        

        plt.subplot(2, 1, 1)

        plt.plot(X0[:5000,:])

        plt.title('ch ' + str(i) + ': preictal absent or class 0')

        

        plt.subplot(2, 1, 2)

        plt.plot(X1[:5000,:])

        plt.title('ch ' + str(i) + ': preictal present or class 1')

        

        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, hspace=0.8)

        plt.show()

        

        plt.subplot(2, 1, 1)

        plt.imshow(IMG0)

        plt.title('ch ' + str(i) + ': preictal absent or class 0')      

                

        plt.subplot(2, 1, 2)

        plt.imshow(IMG1)

        plt.title('ch ' + str(i) + ': preictal present or class 1')

        

        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, hspace=0.8)

        plt.show()
pairs = []

start = 1; stop = 15

for i in range(start,stop):

    pairs.append(['../input/train_1/1_' + str(i) + '_0.mat', 

                  '../input/train_1/1_' + str(i) + '_1.mat'])

ch = N

ch_pairs = []

for i in range(ch):

    for j in range(i+1,ch):

        ch_pairs.append((i,j))



plot_coherence(pairs, ch_pairs)