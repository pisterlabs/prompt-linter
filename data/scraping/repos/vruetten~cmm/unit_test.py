import pytest
from cmm.toy_data import make_toy_data
import numpy as np
from scipy.signal import coherence


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


def test_compute_coherence():
    from cmm.spectral_funcs import compute_coherence

    coherence_yxf, freqs = compute_coherence(
        xnt[:1], xnt, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=False
    )
    sci_freqs, sci_coherence_yxf = coherence(
        xnt[:1], xnt, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=False
    )
    sci_coherence_yxf = sci_coherence_yxf.squeeze()
    assert np.allclose(coherence_yxf, sci_coherence_yxf)


def test():
    from cmm import spectral_funcs as sf
    from cmm.cmm_funcs import compute_spectral_coefs_by_hand

    coefs_xnkf, _ = sf.compute_spectral_coefs(
        xnt, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=False
    )
    coefs_xnkf_hand = compute_spectral_coefs_by_hand(
        xnt, fs=fs, nperseg=nperseg, noverlap=noverlap
    )
    maxdiff = np.abs(coefs_xnkf - coefs_xnkf_hand).max()
    coefs_all_close = np.allclose(coefs_xnkf, coefs_xnkf_hand)
    print(f"\ncoefs_all_close: {coefs_all_close}, max diff: {maxdiff}")
    assert maxdiff < 1e-3
