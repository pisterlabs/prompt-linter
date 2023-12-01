import numpy as np
from cmm import toy_data
import matplotlib.pyplot as pl
from scipy.signal import coherence
from cmm import spectral_funcs as sf
from importlib import reload
from cmm import cmm
from cmm import utils
import jax.numpy as jnp
from cmm import cmm_funcs
from cmm.utils import build_fft_trial_projection_matrices

reload(toy_data)
np.random.seed(4)
path = "/Users/ruttenv/Documents/code/cmm/results/"
t_ = 1000
fs = 10
nperseg = 80
noverlap = int(0.8 * nperseg)
subn = 5
m = 3
tau = 0.1
xnt, ymt, xax, xnkf = toy_data.make_toy_data(
    subn, t_, fs, m, nperseg=nperseg, noverlap=noverlap, noise=0, tau=tau
)
n, t = xnt.shape

ft = nperseg // 2 + 1
k = t // nperseg


Wktf, iWktf = utils.build_fft_trial_projection_matrices(
    t, nperseg=nperseg, noverlap=noverlap, fs=fs
)

xnkf_coefs = np.tensordot(xnt, Wktf, axes=(1, 1))  # sum over t
# pl.figure
# pl.imshow(Wktf[8].real)
# pl.savefig(path + "Wktf")

# xnkt = np.einsum("nkf, tf-> nkt", xnkf, 1 / DTF_tf).real
# print(k, xnkf_coefs.shape[1])
# n, k, f = xnkf_coefs.shape

xnt_ = np.einsum("mkf,ktf->mt", xnkf_coefs, iWktf).real

sl = slice(nperseg * 0, nperseg * 3)
pl.figure()
pl.plot(xax[sl], xnt_[0, sl], label="proj")
pl.plot(xax[sl], xnt[0, sl])
pl.xlabel("time")
pl.legend()
pl.ylabel("backproj_single_trial")
pl.savefig(path + "tmp")

# print(xnkt.shape)

# pl.figure()
# pl.plot(xax, xnt[0])
# pl.xlabel("time")
# pl.ylabel("magnitude")
# pl.savefig(path + "tmp")


# pl.figure()
# pl.plot(freqs, mags_mean_f)
# pl.xlabel("freq [Hz]")
# pl.ylabel("magnitude")
# pl.savefig(path + "generated spectrum")

pxx, freqs = sf.estimate_spectrum(
    xnt,
    fs=fs,
    nperseg=nperseg,
    noverlap=noverlap,
    alltoall=False,
    detrend=False,
    normalize_per_trial=True,
)
# print(freqs.shape, nperseg)
pl.figure()
pl.plot(freqs, pxx[0].real, "o-")
pl.xlabel("freq [Hz]")
pl.ylabel("magnitude")
pl.savefig(path + "spectrum")
# print(pxx.shape)
