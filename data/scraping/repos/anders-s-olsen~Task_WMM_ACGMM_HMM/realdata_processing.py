import sys
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm, trange
from matplotlib import animation
from scipy.signal import hilbert
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid
import h5py
import torch

sys.path.append(os.path.abspath(os.path.join('..')))

from src.preprocessing.bandpass_filter import butter_bandpass_filter
from src.preprocessing.hilbert_phase import hilbert_phase_extract
from src.preprocessing.coherence_LEiDA import coherenceMap, leadingEigenVec

subject = 0
hf = h5py.File('../data/processed/dataset_all_subjects_LEiDA_100.hdf5', 'w')
for file in list(glob.glob('../data/processed/atlas100*.h5')):
    data_subject = h5py.File(file, mode='r')
    org = torch.tensor(np.array(data_subject['data'])).float()

    band = butter_bandpass_filter(org)
    hil1 = hilbert_phase_extract(band)
    maps = coherenceMap(hil1)
    V1 = leadingEigenVec(maps)
    
    subject_id = f'subject_{(subject+1):04d}_LEiDA'
    hf.create_dataset(subject_id, data=V1)
    print('Done with subject '+str(subject))
    subject +=1

