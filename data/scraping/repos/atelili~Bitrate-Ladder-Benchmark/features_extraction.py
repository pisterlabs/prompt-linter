"""
Usage:
    python features_extration -r (raw videos directory) 
    all videos must be YUV420 format
Author : 
    Ahmed Telili
"""



import collections
import argparse
import numpy as np
import scipy.signal as signal
import scipy.special as special
import scipy.optimize as optimize
import skimage.io
import skimage.transform
from tqdm import tqdm
import cv2
import os
from statistics import mean, stdev
import pandas as pd 
from scipy.stats import skew, kurtosis, entropy
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops
from scipy.signal import coherence
from skimage.restoration import estimate_sigma


def image_colorfulness(image):
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(image.astype("float"))
    # compute rg = R - G
    rg = np.absolute(R - G)
    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)
    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)




def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data-mean_data)/(std_data)


def ncc(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    """
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Extract_features")
    parser.add_argument('-r',
        '--raw_dir',
        default='',
        type=str,
        help='Directory path of raw videos')

    args = parser.parse_args()

    videos = os.listdir(args.raw_dir)
    features = []
    for v in tqdm(videos) :
      if (v.endswith('.yuv')) and v.split('_')[1]=='3840x2160':
        video_features = []
        video_path = './' + args.raw_dir +'/'  + v
        file_size = os.path.getsize(video_path)
        names = v.split('_')[0]
        resolution = v.split('_')[1]
        width = int(resolution.split('x')[0])
        height = int(resolution.split('x')[1])
        # Number of frames: in YUV420 frame size in bytes is width*height*1.5
        n_frames = file_size // (width*height*3 // 2)
        correlation = []
        contrast = []
        energy = []
        homogeneity = []
        entrop = []
        TC_mean = []
        TC_std = []
        NCC_f = []
        colf = []
        noise = []
        # Open 'input.yuv' a binary file.
        f = open(video_path, 'rb')

        for i in range(n_frames):
            # Read Y, U and V color channels and reshape to height*1.5 x width numpy array
            yuv = np.frombuffer(f.read(width*height*3//2), dtype=np.uint8).reshape((height*3//2, width))
            
            # Convert YUV420 to BGR (for testing), applies BT.601 "Limited Range" conversion.
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            
            # Convert YUV420 to Grayscale
            gray = cv2.cvtColor(yuv, cv2.COLOR_YUV2GRAY_I420)

            #nss = calculate_brisque_features(gray)

            m = graycomatrix(gray, [5], [0], 256, symmetric=True, normed=True)
            contrast.append(graycoprops(m, prop='contrast')[0,0]) 
            homogeneity.append(graycoprops(m, prop='homogeneity')[0,0])
            energy.append(graycoprops(m, prop='energy')[0,0])
            correlation.append(graycoprops(m, prop='correlation')[0,0])
            entrop.append(shannon_entropy(m))
            colf.append(image_colorfulness(bgr))
            noise.append(estimate_sigma(bgr, multichannel=True, average_sigmas=True))
            if i>0 :

              t, coh = coherence(previous_gray, gray)
              NCC_f.append(ncc(previous_gray,gray))
              TC_mean.append(coh.mean())
              TC_std.append(coh.std())



            
            previous_gray = gray

        TC_mean = np.array(TC_mean)
        TC_std = np.array(TC_std)

        video_features.extend([names, mean(correlation), stdev(correlation), mean(contrast), stdev(contrast), mean(energy), stdev(energy),
                              mean(homogeneity), stdev(homogeneity), mean(entrop), stdev(entrop),TC_mean.mean(), TC_mean.std(),
                               TC_std.mean(), TC_std.std(), skew(TC_mean), skew(TC_std), kurtosis(TC_mean), kurtosis(TC_std), entropy(TC_mean), entropy(TC_std), 
                               mean(NCC_f), stdev(NCC_f), mean(colf), stdev(colf), mean(noise), stdev(noise)])
        features.append(video_features)
        

        f.close()
        
    df = pd.DataFrame(features, columns =['Video', 'Mean_corr', 'Std_corr', 'Mean_cont', 'Std_cont', 'Mean_eng', 'Std_eng', 'Mean_hom', 'Std_hom', 'Mean_entr',
                                          'Std_entr', 'TC_mean_mean', 'TC_mean_std', 'TC_std_mean', 'TC_std_std', 'TC_mean_skew', 'TC_std_skew','TC_mean_kurtosis','TC_std_kurtosis',
                                          'TC_mean_entr', 'TC_std_entr', 'NCC_mean', 'NCC_std','colorfulness_mean', 'colorfulness_std','Noise_mean','Noise_std'])
    df.to_csv('features.csv')  
