




import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import csv2rec
import scipy.stats.distributions as dist
from scipy import fftpack

import nitime
from nitime.timeseries import TimeSeries
from nitime import utils
import nitime.algorithms as alg
import nitime.viz
from nitime.viz import drawmatrix_channels
from nitime.analysis import CoherenceAnalyzer, MTCoherenceAnalyzer

TR = 1.89
f_ub = 0.15
f_lb = 0.02

data_path = os.path.join(nitime.__path__[0], 'data')

data_rec = csv2rec(os.path.join(data_path, 'fmri_timeseries.csv'))

roi_names = np.array(data_rec.dtype.names)
nseq = len(roi_names)
n_samples = data_rec.shape[0]
print (data_rec.shape)
print (nseq)
data = np.zeros((nseq, n_samples))

for n_idx, roi in enumerate(roi_names):
    data[n_idx] = data_rec[roi]

pdata = utils.percent_change(data)
print(pdata.shape)

NW = 5
K = 2 * NW - 1

tapers, eigs = alg.dpss_windows(n_samples, NW, K)

print (tapers.shape)
print (eigs.shape)

tdata = tapers[None, :, :] * pdata[:, None, :]
print(tdata.shape)

tspectra = fftpack.fft(tdata)
print (tspectra.shape)

L = n_samples // 2 + 1
sides = 'onesided'

w = np.empty((nseq, K, L))
for i in range(nseq):
    w[i], _ = utils.adaptive_weights(tspectra[i], eigs, sides=sides)

print (w[6,8,:])
