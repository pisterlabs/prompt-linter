from cmm import utils
from cmm import spectral_funcs as sf
from cmm.cmm_funcs import compute_spectral_coefs_by_hand
import matplotlib.pyplot as pl
import numpy as np
import matplotlib as mpl
from jax.lax import scan
import jax.numpy as jnp
from cmm.toy_data import make_toy_data
from cmm.cmm import compute_cluster_mean
from cmm import cmm
from scipy.signal import coherence

np.random.seed(4)
mpl.pyplot.rcParams.update({"text.usetex": True})
path = "/Users/ruttenv/Documents/code/cmm/results/"

rpath = path + "res.npy"

t_ = 800
fs = 20
nperseg = 80
noverlap = int(0.8 * nperseg)
subn = 6
m = 3
freq_minmax = [0, 3]
freq_minmax = [-np.inf, np.inf]
noise = 1e-4
tau = 0.1
#######################
### make toy data
xnt, ymt, xax, xknf = make_toy_data(
    subn, t_, fs, m, nperseg=nperseg, noverlap=noverlap, noise=noise, tau=tau
)


###################### compute coherence from time series
from cmm.spectral_funcs import compute_coherence

coherence_yxf, freqs = compute_coherence(
    xnt[:1], xnt, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=False
)
sci_freqs, sci_coherence_yxf = coherence(
    xnt[:1], xnt, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=False
)
sci_coherence_yxf = sci_coherence_yxf.squeeze()
allclose = np.allclose(coherence_yxf, sci_coherence_yxf)
print(f"coherence allclose: {allclose}")


######## compute coefs using standard method vs matrix multiplication

coefs_xnkf, _ = sf.compute_spectral_coefs(
    xnt, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=False
)
coefs_xnkf_hand = compute_spectral_coefs_by_hand(
    xnt, fs=fs, nperseg=nperseg, noverlap=noverlap
)
maxdiff = np.abs(coefs_xnkf - coefs_xnkf_hand).max()
coefs_all_close = np.allclose(coefs_xnkf, coefs_xnkf_hand)
print(f"\ncoefs_all_close: {coefs_all_close}, max diff: {maxdiff}")

coefs_ymkf_hand = compute_spectral_coefs_by_hand(
    ymt, fs=fs, nperseg=nperseg, noverlap=noverlap
)

######## compute coherence using matrix multiplication extracted coeficients
coherence_yxf_byhand, freqs = compute_coherence(
    coefs_xnkf_hand[:1],
    coefs_xnkf_hand,
    fs=fs,
    nperseg=nperseg,
    noverlap=noverlap,
    detrend=False,
    x_in_coefs=True,
    y_in_coefs=True,
)
coherence_yxf_byhand = coherence_yxf_byhand.squeeze()
coh_by_hand_allclose = np.allclose(coherence_yxf_byhand, sci_coherence_yxf)
maxdiff = np.abs(coherence_yxf_byhand - sci_coherence_yxf).max()
print(f"\ncoh_by_hand_allclose: {coh_by_hand_allclose}, max diff: {maxdiff}")


### check if eigenvalues are equivalent to square of sum of coherences
n, t = xnt.shape
print(f"n t: {n, t}")
opt_in_freqdom = False
opt_in_freqdom = True
k = m + 1
cm = cmm.CMM(
    xnt,
    k=k,
    fs=fs,
    nperseg=nperseg,
    noverlap=noverlap,
    freq_minmax=freq_minmax,
    opt_in_freqdom=opt_in_freqdom,
)


topn = 5
mu_fk, eigvals_f = compute_cluster_mean(
    coefs_xnkf_hand[:topn],
    nperseg=nperseg,
    noverlap=noverlap,
    fs=fs,
    x_in_coefs=True,
    return_temporal_proj=False,
)

coherence_mnk = cm.compute_cross_coherence_from_coefs(
    mu_fk.T[None], coefs_xnkf_hand[:topn]
)

top = 3
print((coherence_mnk[:1, :, :top]).sum(1), eigvals_f[:top])

#### compute coherence by hand
y1k = mu_fk[0][None]
xnk = coefs_xnkf_hand[:topn, :, 0]

xnk_ = xnk / np.sqrt((xnk * np.conj(xnk)).sum(-1)[:, None])
coh = (np.abs((y1k * np.conj(xnk_)).sum(-1)) ** 2).sum()
print(coh)

### result proven: eigenvalue is simply sum of the coherences
