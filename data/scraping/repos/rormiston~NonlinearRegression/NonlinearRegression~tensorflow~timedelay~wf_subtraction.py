import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import noisesub as n
import os
# Hush TF AVX warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '5'
import NonlinearRegression.tools.preprocessing as ppr
import argparse

def parse_command_line():
    """
    parse command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", "-s",
                        help    = "channel start",
                        default = 1,
                        dest    = "start",
                        type    = int)

    parser.add_argument("--end", "-e",
        				help    = "channel end",
                        default = 2,
        				dest    = "end",
                        type    = int)
    params = parser.parse_args()
    return params

params = parse_command_line()
start = params.start
end = params.end

dataset, fs = ppr.get_dataset('Data/L1_data_array.mat', data_type='real')
target = dataset[:, 0].T
witness = dataset[:, 2:].T
t_fft = 8

##################################
#fs = 1e3
#t_total = 100
#t_fft = 3
#L = int(fs*t_total)
## Random data streams
#target_background = np.random.randn(L)
#witness_background = np.random.randn(L)
#coupled = 10*np.random.randn(L)

## Lowpass for some non-flat shape of coupling
#bb, aa = sig.butter(4, 0.05)

#target = sig.lfilter(bb, aa, coupled) + target_background
#witness = coupled + witness_background  # Witness is imperfect
######################################

# Use coherence to estimate subtraction potential
psd_kwargs = {'fs': fs, 'nperseg':t_fft*fs}
ff, C = sig.coherence(target, witness, **psd_kwargs)

# Train wiener filter
t_impulse = 1.0  # Length, in seconds, of FIR filter kernel
W = n.wiener_fir(target, witness, t_impulse*fs)
prediction = sig.lfilter(W[0], 1.0, witness)
subtracted = target - prediction

# Calculate some PSDs, plot

_, p_target = sig.welch(target, **psd_kwargs)
# _, p_background = sig.welch(target_background, **psd_kwargs)
_, p_subtracted = sig.welch(subtracted, **psd_kwargs)

plt.figure()
plt.loglog(ff, np.sqrt(p_target), label='Subtraction Target')
plt.loglog(ff, np.sqrt(p_subtracted).T, label='Wiener Filter Subtraction')
plt.loglog(ff, np.sqrt(p_target*(1-C)).T, label='Prediction from Coherence')
# plt.loglog(ff, np.sqrt(p_background), label='True Background')
plt.grid(True, which='both')
plt.axis('tight')
plt.xlim([7, 150])
plt.ylim([5e-13, 1e-8])
plt.xlabel('Freq [Hz.]')
plt.ylabel('ASD [arb/rtHz]')
plt.title('Wiener Filtering example')
plt.legend()
plt.savefig('WF_example.png')
