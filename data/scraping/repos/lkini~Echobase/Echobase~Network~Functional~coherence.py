"""
Correlation-based measure for computing functional connectivity

Created by: Ankit Khambhati

Change Log
----------
2016/03/06 - Implemented WelchCoh and MTCoh pipes
"""

from __future__ import division
import numpy as np
from mtspec import mt_coherence, mtspec
from scipy.signal import coherence

from ...Common import errors


def multitaper(data, fs, time_band, n_taper, cf):
    """
    The multitaper function windows the signal using multiple Slepian taper
    functions and then computes coherence between windowed signals.

    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates

        fs: int
            Sampling frequency

        time_band: float
            The time half bandwidth resolution of the estimate [-NW, NW];
            such that resolution is 2*NW

        n_taper: int
            Number of Slepian sequences to use (Usually < 2*NW-1)

        cf: list
            Frequency range over which to compute coherence [-NW+C, C+NW]

    Returns
    -------
        adj: ndarray, shape (N, N)
            Adjacency matrix for N variates
    """

    # Standard param checks
    errors.check_type(data, np.ndarray)
    errors.check_dims(data, 2)
    errors.check_type(fs, int)
    errors.check_type(time_band, float)
    errors.check_type(n_taper, int)
    errors.check_type(cf, list)
    if n_taper >= 2*time_band:
        raise Exception('Number of tapers must be less than 2*time_band')
    if not len(cf) == 2:
        raise Exception('Must give a frequency range in list of length 2')

    # Get data attributes
    n_samp, n_chan = data.shape
    triu_ix, triu_iy = np.triu_indices(n_chan, k=1)

    # Initialize adjacency matrix
    adj = np.zeros((n_chan, n_chan))

    # Compute all coherences
    for n1, n2 in zip(triu_ix, triu_iy):
        out = mt_coherence(1.0/fs,
                           data[:, n1],
                           data[:, n2],
                           time_band,
                           n_taper,
                           int(n_samp/2.), 0.95,
                           iadapt=1,
                           cohe=True, freq=True)

        # Find closest frequency to the desired center frequency
        cf_idx = np.flatnonzero((out['freq'] >= cf[0]) &
                                (out['freq'] <= cf[1]))

        # Store coherence in association matrix
        adj[n1, n2] = np.mean(out['cohe'][cf_idx])
    adj += adj.T

    return adj
