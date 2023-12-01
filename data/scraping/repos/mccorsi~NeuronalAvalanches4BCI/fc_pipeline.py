"""
=================================
Alternative features based on dedicated MEG dataset
=================================
This module is design to compute neuronal avalanches metrics on a MEG dataset
"""
# Authors: Pierpaolo Sorrentino, Marie-Constance Corsi
#
# License: BSD (3-clause)
import mne.time_frequency
from sklearn.covariance import ledoit_wolf
from sklearn.base import BaseEstimator, TransformerMixin

import hashlib
import os.path as osp
import os

from mne import get_config, set_config, set_log_level, EpochsArray
from mne.connectivity import spectral_connectivity
from mne.connectivity import envelope_correlation
from mne.time_frequency import psd_multitaper
from moabb.evaluations.base import BaseEvaluation
from scipy.stats import zscore

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from time import time
import numpy as np
from mne.epochs import BaseEpochs
from sklearn.metrics import get_scorer

from pyriemann.classification import MDM, FgMDM
from pyriemann.estimation import Coherences

from scipy import stats as spstats


#%% functional connectivity

def _compute_fc_subtrial(epoch, delta=1, ratio=0.5, method="coh", fmin=8, fmax=35):
    """Compute single trial functional connectivity (FC)

    Most of the FC estimators are already implemented in mne-python (and used here from
    mne.connectivity.spectral_connectivity and mne.connectivity.envelope_correlation).
    The epoch is split into subtrials.

    Parameters
    ----------
    epoch: MNE epoch
        Epoch to process
    delta: float
        length of the subtrial in seconds
    ratio: float, in [0, 1]
        ratio overlap of the sliding windows
    method: string
        FC method to be applied, currently implemented methods are: "coh", "plv",
        "imcoh", "pli", "pli2_unbiased", "wpli", "wpli2_debiased", "cov", "plm", "aec"
    fmin: real
        filtering frequency, lowpass, in Hz
    fmax: real
        filtering frequency, highpass, in Hz

    Returns
    -------
    connectivity: array, (nb channels x nb channels)

    #TODO: compare matlab/python plm's output
    The only exception is the Phase Linearity Measurement (PLM). In this case, it is a
    Python version of the ft_connectivity_plm MATLAB code [1] of the Fieldtrip
    toolbox [2], which credits [3], with the "translation" into Python made by
    M.-C. Corsi.

    references
    ----------
    .. [1] https://github.com/fieldtrip/fieldtrip/blob/master/connectivity/ft_connectivity_plm.m  # noqa
    .. [2] R. Oostenveld, P. Fries, E. Maris, J.-M. Schoffelen, and  R. Oostenveld,
    "FieldTrip: Open Source Software for Advanced Analysis of MEG, EEG, and Invasive
    Electrophysiological  Data" (2010): https://doi.org/10.1155/2011/156869
    .. [3] F. Baselice, A. Sorriso, R. Rucco, and P. Sorrentino, "Phase Linearity
    Measurement: A Novel Index for Brain Functional Connectivity" (2019):
    https://doi.org/10.1109/TMI.2018.2873423
    """
    lvl = set_log_level("CRITICAL")
    L = epoch.times[-1] - epoch.times[0]
    sliding = ratio * delta
    # fmt: off
    spectral_met = ["coh", "plv", "imcoh", "pli", "pli2_unbiased",
                    "wpli", "wpli2_debiased", ]
    other_met = ["cov", "plm", "aec"]
    # fmt: on
    if not method in spectral_met + other_met:
        raise NotImplemented("this spectral connectivity method is not implemented")

    sfreq, nb_chan = epoch.info["sfreq"], epoch.info["nchan"]
    win = delta * sfreq
    nb_subtrials = int(L * (1 / (sliding + delta) + 1 / delta))
    nbsamples_subtrial = delta * sfreq

    # TODO:
    #  - reboot computation options depending on the frequency options, faveage=False, but issue on AEC /PLM :/
    #  - robust estimators : bootstrap over subtrials, sub-subtrials & z-score, ways to remove outliers

    # X, total nb trials over the session(s) x nb channels x nb samples
    X = np.squeeze(epoch.get_data())
    subtrials = np.empty((nb_subtrials, nb_chan, int(win)))

    for i in range(0, nb_subtrials):
        idx_start = int(sfreq * i * sliding)
        idx_stop = int(sfreq * i * sliding + nbsamples_subtrial)
        subtrials[i, :, :] = np.expand_dims(X[:, idx_start:idx_stop], axis=0)
    sub_epoch = EpochsArray(np.squeeze(subtrials), info=epoch.info)
    if method in spectral_met:
        r = spectral_connectivity(
            sub_epoch,
            method=method,
            mode="multitaper",
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            tmin=0,
            mt_adaptive=False,
            n_jobs=1,
        )
        c = np.squeeze(r[0])
        c = c + c.T - np.diag(np.diag(c)) + np.identity(nb_chan)
    elif method == "aec":
        # filter in frequency band of interest
        sub_epoch.filter(
            fmin,
            fmax,
            n_jobs=1,
            l_trans_bandwidth=1,  # make sure filter params are the same
            h_trans_bandwidth=1,
        )  # in each band and skip "auto" option.
        # apply hilbert transform first
        h_sub_epoch = sub_epoch.apply_hilbert()
        c = envelope_correlation(h_sub_epoch, verbose=True)
        # by default, combine correlation estimates across epochs by peforming an average
        # output : nb_channels x nb_channels -> no need to rearrange the matrix
    elif method == "cov":
        c = ledoit_wolf(X.T)[0]  # oas ou fast_mcd
    elif method == "plm":
        # adapted from the matlab code from https://github.com/fieldtrip/fieldtrip/blob/master/connectivity/ft_connectivity_plm.m  # noqa
        # TODO: compare matlab/python plm's output
        # no need to filter before because integration in the frequency band later on

        # apply hilbert transform first
        h_sub_epoch = sub_epoch.apply_hilbert()
        input = h_sub_epoch.get_data()

        ph_min = 0.1  # Eps of Eq.(17) of the manuscript
        f = [
            int(sfreq / nbsamples_subtrial) * x
            for x in range(int(nbsamples_subtrial) - 1)
        ]
        # B: bandwidth, in hertz
        B = fmax - fmin
        f_diff = np.zeros((len(f), 1))
        for i in range(len(f)):
            f_diff[i] = f[i] - sfreq
        idx_f_integr_temp = np.where(
            np.logical_and(np.abs(f) < B, np.abs(f_diff) < B) == True
        )
        idx_f_integr = idx_f_integr_temp[1]

        p = np.zeros((nb_chan, nb_chan, len(input)))
        for i in range(len(input)):
            for kchan1 in range(nb_chan - 2):
                for kchan2 in range((kchan1 + 1), nb_chan):
                    temp = np.fft.fft(
                        input[i, kchan1, :] * np.conjugate(input[i, kchan2, :])
                    )
                    temp[0] = temp[0] * (abs(np.angle(temp[0])) > ph_min)
                    # TODO: check temp values, they are really low
                    temp = np.power((abs(temp)), 2)
                    p_temp = np.sum(temp[idx_f_integr,]) / np.sum(temp)
                    p[kchan1, kchan2, i] = p_temp
                    p[kchan2, kchan1, i] = p_temp
                    # to make it symmetrical i guess
        # new, not in the matlab code, average over the
        # subtrials + normalization:
        m = np.mean(p, axis=2)
        c1 = m / np.max(m) + np.identity(nb_chan)
        c = np.moveaxis(c1, -1, 0)
    return c


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def isPD2(B):
    """Returns true when input is positive-definite, via eigenvalues"""
    if np.any(np.linalg.eigvals(B) < 0.0):
        return False
    else:
        return True


def nearestPD(A, reg=1e-6):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): htttps://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        # Regularize if already PD
        ei, ev = np.linalg.eigh(A3)
        if np.min(ei) / np.max(ei) < reg:
            A3 = ev @ np.diag(ei + reg) @ ev.T
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD2(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    # Regularize
    ei, ev = np.linalg.eigh(A3)
    if np.min(ei) / np.max(ei) < reg:
        A3 = ev @ np.diag(ei + reg) @ ev.T
    return A3


class FunctionalTransformer(TransformerMixin, BaseEstimator):
    """Getting connectivity features from epoch"""

    def __init__(self, delta=1, ratio=0.5, method="coh", fmin=8, fmax=35):
        self.delta = delta
        self.ratio = ratio
        self.method = method
        self.fmin = fmin
        self.fmax = fmax
        if get_config("MOABB_PREPROCESSED") is None:
            set_config(
                "MOABB_PREPROCESSED",
                osp.join(osp.expanduser("~"), "mne_data", "preprocessing"),
            )
        if not osp.isdir(get_config("MOABB_PREPROCESSED")):
            os.makedirs(get_config("MOABB_PREPROCESSED"))
        self.preproc_dir = get_config("MOABB_PREPROCESSED")
        self.cname = "-".join(
            [
                str(e)
                for e in [
                self.method,
                self.delta,
                self.ratio,
                self.fmin,
                self.fmax,
                ".npz",
            ]
            ]
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # StackingClassifier uses cross_val_predict, that apply transform
        # with dispatch_one_batch, streaming each trial one by one :'(
        # If training on a whole set, cache results otherwise compute
        # fc each time
        if isinstance(X, BaseEpochs):
            if self.method in ['instantaneous', 'lagged']:
                Xfc_temp = Coherences(coh=self.method, fmin=self.fmin, fmax=self.fmax,
                                      fs=X.info["sfreq"]).fit_transform(X.get_data())
                Xfc = np.empty(Xfc_temp.shape[:-1], dtype=Xfc_temp.dtype)
                for trial, fc in enumerate(Xfc_temp):
                    Xfc[trial, :, :] = fc.mean(axis=-1)
                return Xfc

            fcache = hashlib.md5(X.get_data()).hexdigest() + self.cname
            # fcache = osp.join(
            #     "./", fcache
            # )  # line changed because self.preproc_dir leads to mne_data folder ^^'
            if osp.isfile(fcache):
                return np.load(fcache)["Xfc"]
            else:
                Xfc = np.empty((len(X), X[0].info["nchan"], X[0].info["nchan"]))
                for i in range(len(X)):
                    Xfc[i, :, :] = _compute_fc_subtrial(
                        X[i],
                        delta=self.delta,
                        ratio=self.ratio,
                        method=self.method,
                        fmin=self.fmin,
                        fmax=self.fmax,
                    )
                # np.savez_compressed(fcache, Xfc=Xfc)

            return Xfc

#%% neuronal avalanches

def transprob(aval,nregions): # (t,r)
    mat = np.zeros((nregions, nregions))
    norm = np.sum(aval, axis=0)
    for t in range(len(aval) - 1):
        ini = np.where(aval[t] == 1)
        mat[ini] += aval[t + 1]
    mat[norm != 0] = mat[norm != 0] / norm[norm != 0][:, None]
    return mat

def Transprob(ZBIN,nregions): # (t,r)
    mat = np.zeros((nregions, nregions))
    A = np.sum(ZBIN, axis=1)
    a = np.arange(len(ZBIN))
    idx = np.where(A != 0)[0]
    aout = np.split(a[idx], np.where(np.diff(idx) != 1)[0] + 1)
    #print(aout,np.shape(aout))
    ifi = 0
    for iaut in range(len(aout)):
        if len(aout[iaut]) > 2: # min size aval = 2 ???
            mat += transprob(ZBIN[aout[iaut]],nregions)
            ifi += 1
    mat = mat / ifi
    return mat,aout

def threshold_mat(data,thresh=3):
    current_data=data
    binarized_data=np.where(np.abs(current_data)>thresh,1,0)
    return (binarized_data)

def find_avalanches(data,thresh=3):
    binarized_data=threshold_mat(data,thresh=thresh)
    N=binarized_data.shape[0]
    mat, aout = Transprob(binarized_data.T, N)
    aout=np.array(aout,dtype=object)
    list_length=[len(i) for i in aout]
    unique_sizes=set(list_length)
    min_size,max_size=min(list_length),max(list_length)
    list_avalanches_bysize={i:[] for i in unique_sizes}
    for s in aout:
        n=len(s)
        list_avalanches_bysize[n].append(s)
    return(aout,min_size,max_size,list_avalanches_bysize, mat)

class ATMTransformer(TransformerMixin, BaseEstimator):
    """Getting Avalanche Transition Matrices (ATMs) features from epoch"""
    def __init__(self, zthresh=3):
        self.zthresh = zthresh
        if get_config("MOABB_PREPROCESSED") is None:
            set_config(
                "MOABB_PREPROCESSED",
                osp.join(osp.expanduser("~"), "mne_data", "preprocessing"),
            )
        if not osp.isdir(get_config("MOABB_PREPROCESSED")):
            os.makedirs(get_config("MOABB_PREPROCESSED"))
        self.preproc_dir = get_config("MOABB_PREPROCESSED")
        self.cname = "-".join(
            [
                str(e)
                for e in [
                self.zthresh,
                ".npz",
            ]
            ]
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, BaseEpochs):
            data = np.squeeze(X.get_data())
            ATM=np.empty((len(X),68,68))
            for kk_trial in range(len(X)):
                zscored_data = zscore(data[kk_trial,:,:], axis=1)
                list_avalanches,min_size_avalanches,max_size_avalanches,list_avalanches_bysize,temp_ATM=find_avalanches(zscored_data,thresh=self.zthresh)
                #ATM: nb_trials x nb_ROIs x nb_ROIs matrix
                ATM[kk_trial,:,:]=temp_ATM
                # # transform matrix to array to apply csp/lda?
                # array_ATM = np.squeeze(np.array(ATM))
            return ATM

#%% power spectra
class PowerTransformer(TransformerMixin, BaseEstimator):
    """Getting Power spectra (from multitaper) features from epoch"""
    def __init__(self, fmin=8, fmax=35):
        self.fmin = fmin
        self.fmax=fmax
        if get_config("MOABB_PREPROCESSED") is None:
            set_config(
                "MOABB_PREPROCESSED",
                osp.join(osp.expanduser("~"), "mne_data", "preprocessing"),
            )
        if not osp.isdir(get_config("MOABB_PREPROCESSED")):
            os.makedirs(get_config("MOABB_PREPROCESSED"))
        self.preproc_dir = get_config("MOABB_PREPROCESSED")
        self.cname = "-".join(
            [
                str(e)
                for e in [
                self.fmin,
                self.fmax,
                ".npz",
            ]
            ]
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, BaseEpochs):
            psds, freqs = psd_multitaper(inst=X,fmin=self.fmin, fmax=self.fmax)
            #transform matrix to array to apply csp/lda?
            #array_psds=np.squeeze(np.array(psds))
            # average over all the freq band (default 8-35)
            power=np.squueze(np.mean(psds,2)) # psds: n_epochs x n_channels x n_freqs
            return power # n_epochs x n_channels

#%% other ####################################
class EnsureSPD(TransformerMixin, BaseEstimator):
    """Getting connectivity features from mat files"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xspd = np.empty_like(X)
        for i, mat in enumerate(X):
            Xspd[i, :, :] = nearestPD(mat)
        return Xspd

    def fit_transform(self, X, y=None):
        transf = self.transform(X)
        return transf

#####################################
class GetMEGdata(TransformerMixin, BaseEstimator):
    """Getting MEG data from epochs"""
    ch_names = ['MEG_bankssts L',
                'MEG_bankssts R',
                'MEG_caudalanteriorcingulate L',
                'MEG_caudalanteriorcingulate R',
                'MEG_caudalmiddlefrontal L',
                'MEG_caudalmiddlefrontal R',
                'MEG_cuneus L',
                'MEG_cuneus R',
                'MEG_entorhinal L',
                'MEG_entorhinal R',
                'MEG_frontalpole L',
                'MEG_frontalpole R',
                'MEG_fusiform L',
                'MEG_fusiform R',
                'MEG_inferiorparietal L',
                'MEG_inferiorparietal R',
                'MEG_inferiortemporal L',
                'MEG_inferiortemporal R',
                'MEG_insula L',
                'MEG_insula R',
                'MEG_isthmuscingulate L',
                'MEG_isthmuscingulate R',
                'MEG_lateraloccipital L',
                'MEG_lateraloccipital R',
                'MEG_lateralorbitofrontal L',
                'MEG_lateralorbitofrontal R',
                'MEG_lingual L',
                'MEG_lingual R',
                'MEG_medialorbitofrontal L',
                'MEG_medialorbitofrontal R',
                'MEG_middletemporal L',
                'MEG_middletemporal R',
                'MEG_paracentral L',
                'MEG_paracentral R',
                'MEG_parahippocampal L',
                'MEG_parahippocampal R',
                'MEG_parsopercularis L',
                'MEG_parsopercularis R',
                'MEG_parsorbitalis L',
                'MEG_parsorbitalis R',
                'MEG_parstriangularis L',
                'MEG_parstriangularis R',
                'MEG_pericalcarine L',
                'MEG_pericalcarine R',
                'MEG_postcentral L',
                'MEG_postcentral R',
                'MEG_posteriorcingulate L',
                'MEG_posteriorcingulate R',
                'MEG_precentral L',
                'MEG_precentral R',
                'MEG_precuneus L',
                'MEG_precuneus R',
                'MEG_rostralanteriorcingulate L',
                'MEG_rostralanteriorcingulate R',
                'MEG_rostralmiddlefrontal L',
                'MEG_rostralmiddlefrontal R',
                'MEG_superiorfrontal L',
                'MEG_superiorfrontal R',
                'MEG_superiorparietal L',
                'MEG_superiorparietal R',
                'MEG_superiortemporal L',
                'MEG_superiortemporal R',
                'MEG_supramarginal L',
                'MEG_supramarginal R',
                'MEG_temporalpole L',
                'MEG_temporalpole R',
                'MEG_transversetemporal L',
                'MEG_transversetemporal R']

    chan_meg = [i for i in ch_names if i.startswith('MEG')]

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, chan_meg):
        if isinstance(X, BaseEpochs):
            meg = X.pick_channels(chan_meg)
            return meg

    def fit_transform(self, X, y=None):
        transf = self.transform(X)
        return transf


class Snitch(TransformerMixin, BaseEstimator):
    """Getting connectivity features from mat files"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"t: X={X.shape}")
        return X

    def fit_transform(self, X, y=None):
        print(f"ft: X={X.shape}")
        return X


class AvgFC(TransformerMixin, BaseEstimator):
    """Getting connectivity features from mat files"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xavg = np.empty(X.shape[:-1], dtype=X.dtype)
        for trial, fc in enumerate(X):
            Xavg[trial, :, :] = fc.mean(axis=-1)
        return Xavg

    def fit_transform(self, X, y=None):
        return self.transform(X)


class GetData(TransformerMixin, BaseEstimator):
    """Get data for ensemble"""

    def __init__(self, paradigm, dataset, subject):
        self.paradigm = paradigm
        self.dataset = dataset
        self.subject = subject

    def fit(self, X, y=None):
        self.ep_, _, self.metadata_ = self.paradigm.get_data(
            self.dataset, [self.subject], return_epochs=True
        )
        return self

    def transform(self, X):
        return self.ep_[X]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class GetData_ATM(TransformerMixin, BaseEstimator):
    """Get data for ensemble"""

    def __init__(self, paradigm, dataset, subject):
        self.paradigm = paradigm
        self.dataset = dataset
        self.subject = subject

    def fit(self, X, y=None):
        self.ep_, _, self.metadata_ = self.paradigm.get_data(
            self.dataset, [self.subject], return_epochs=False
        )
        return self

    def transform(self, X):
        return self.ep_[X]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class GetData_array(TransformerMixin, BaseEstimator):
    """Get data epochs -> array"""
    def __init__(self, paradigm, dataset, subject):
        self.paradigm = paradigm
        self.dataset = dataset
        self.subject = subject

    def fit(self, X, y=None):
        self.ep_, _, self.metadata_ = self.paradigm.get_data(
            self.dataset, [self.subject], return_epochs=True
        )
        self.ep_data=self.ep_.get_data()
        return self

    def transform(self, X):
        return self.ep_data[X]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class GetDataMemoryMEG(TransformerMixin, BaseEstimator):
    """Get data """

    def __init__(self, freqband, method, precomp_data):
        self.freqband = freqband
        self.method = method
        self.precomp_data = precomp_data

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.precomp_data[self.freqband][self.method][X]

    def fit_transform(self, X, y=None):
        return self.transform(X)

class GetDataMemoryEEG(TransformerMixin, BaseEstimator):
    """Get data """

    def __init__(self, freqband, method, precomp_data):
        self.freqband = freqband
        self.method = method
        self.precomp_data = precomp_data

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.precomp_data[self.freqband][self.method][X]

    def fit_transform(self, X, y=None):
        return self.transform(X)



class GetDataMemoryATM(TransformerMixin, BaseEstimator):
    """Get data """

    def __init__(self, method, precomp_data):
        self.method = method
        self.precomp_data = precomp_data

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.precomp_data

    def fit_transform(self, X, y=None):
        return self.transform(X)


class GetDataMemoryPow(TransformerMixin, BaseEstimator):
    """Get data """

    def __init__(self, method, precomp_data):
        self.method = method
        self.precomp_data = precomp_data

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.precomp_data

    def fit_transform(self, X, y=None):
        return self.transform(X)

#%% Dimensionality reduction ####################################
class FC_DimRed(TransformerMixin, BaseEstimator):
    """Returns the best (threshold, nb_nodes) configuration from X= FC matrices to perform dimension reduction"""

    def __init__(
            self, threshold, classifier=FgMDM(metric="riemann", tsupdate=False), save_ch_fname=None
    ):
        self.threshold = threshold
        # self.nb_nodes = nb_nodes
        self.classifier = classifier
        self.save_ch_fname = save_ch_fname  # if None, don't save, otherwise save selected channel names in fname
        self.best_acc_ = 0

    def fit(self, X, y=None):
        from sklearn.model_selection import cross_val_score

        y0, y1 = np.unique(y)
        idx_0 = np.where(y == y0)
        idx_1 = np.where(y == y1)

        # t-test FC
        FC_right = X[idx_0, :, :].squeeze()
        FC_left = X[idx_1, :, :].squeeze()

        if len(FC_left) < len(FC_right):
            FC_right = FC_right[: len(FC_left), :, :]
        elif len(FC_right) < len(FC_left):
            FC_left = FC_left[: len(FC_right), :, :]
        [self.stats_, self.pvalue_] = spstats.ttest_rel(FC_right, FC_left, axis=0)

        # identify the best configuration (threshold, nb_nodes)
        for th in self.threshold:
            # for n in self.nb_nodes:
            thresh_mask = np.where(self.pvalue_ < th, 1, 0)
            # node_strength_discrim = np.sum(thresh_mask, axis=0)
            # idx = np.argsort(node_strength_discrim)
            # n=len(idx)
            # node_select = np.sort(idx[:n])
            X_temp = X*thresh_mask
            node_select=thresh_mask
            n=np.sum(np.sum(thresh_mask))
            X2use=np.reshape(X_temp,(np.shape(X_temp)[0],np.shape(X_temp)[1]*np.shape(X_temp)[2]))
            scores = cross_val_score(self.classifier, X2use, y, cv=5)

            #if scores.mean() > self.best_acc_:
            self.best_acc_ = scores.mean()
            self.best_param_ = (th, n)
            self.node_select_ = node_select

        if self.best_acc_ == 0: # to take into account all the channels
            th = 1  # to take into account all the channels
            thresh_mask = np.where(self.pvalue_ < th, 1, 0)
            # node_strength_discrim = np.sum(thresh_mask, axis=0)
            # idx = np.argsort(node_strength_discrim)
            # n=len(idx)
            # node_select = np.sort(idx[:n])
            X_temp = X*thresh_mask
            node_select=thresh_mask
            n=np.sum(np.sum(thresh_mask))
            X2use=np.reshape(X_temp,(np.shape(X_temp)[0],np.shape(X_temp)[1]*np.shape(X_temp)[2]))
            scores = cross_val_score(self.classifier, X2use, y, cv=5)

            # if scores.mean() > self.best_acc_:
            self.best_acc_ = scores.mean()
            self.best_param_ = (th, n)
            self.node_select_ = node_select

        if self.save_ch_fname is not None:
            np.savez_compressed(self.save_ch_fname, node_select=self.node_select_, param=self.best_param_,
                                acc=self.best_acc_)


        return self

    def transform(self, X, y=None):
        # to check that we actually perform a DR
        # print(
        # "#################\n" +
        # f"List of channels selected: {self.node_select_}\n" +
        # f"Number of selected channels: {len(self.node_select_)}\n" +
        # f"Best parameters: {self.best_param_}\n" +
        # f"X shape: {X[:, self.node_select_, :][:, :, self.node_select_].shape}\n" +
        # "#############"
        # )

        #preselect_matrix=X[:, self.node_select_, :][:, :, self.node_select_]
        preselect_matrix=X*self.node_select_
        preselect_feat_svm=np.reshape(preselect_matrix,(np.shape(preselect_matrix)[0],np.shape(preselect_matrix)[1]*np.shape(preselect_matrix)[2]))
        return preselect_feat_svm

#%% Class for ATM

def transprob(aval,nregions): # (t,r)
    mat = np.zeros((nregions, nregions))
    norm = np.sum(aval, axis=0)
    for t in range(len(aval) - 1):
        ini = np.where(aval[t] == 1)
        mat[ini] += aval[t + 1]
    mat[norm != 0] = mat[norm != 0] / norm[norm != 0][:, None]
    return mat

def Transprob(ZBIN,nregions, val_duration): # (t,r)
    mat = np.zeros((nregions, nregions))
    A = np.sum(ZBIN, axis=1)
    a = np.arange(len(ZBIN))
    idx = np.where(A != 0)[0]
    aout = np.split(a[idx], np.where(np.diff(idx) != 1)[0] + 1)
    #print(aout,np.shape(aout))
    ifi = 0
    for iaut in range(len(aout)):
        if len(aout[iaut]) > val_duration: # TODO: before 2, tested with 4 too
            mat += transprob(ZBIN[aout[iaut]],nregions)
            ifi += 1
    mat = mat / ifi
    return mat,aout

def threshold_mat(data,thresh=3):
    current_data=data
    binarized_data=np.where(np.abs(current_data)>thresh,1,0)
    return (binarized_data)

def find_avalanches(data,thresh=3, val_duration=2):
    binarized_data=threshold_mat(data,thresh=thresh)
    N=binarized_data.shape[0]
    mat, aout = Transprob(binarized_data.T, N, val_duration)
    aout=np.array(aout,dtype=object)
    list_length=[len(i) for i in aout]
    unique_sizes=set(list_length)
    min_size,max_size=min(list_length),max(list_length)
    list_avalanches_bysize={i:[] for i in unique_sizes}
    for s in aout:
        n=len(s)
        list_avalanches_bysize[n].append(s)
    return(aout,min_size,max_size,list_avalanches_bysize, mat)

class ATMTransformer(TransformerMixin, BaseEstimator):
    """Getting ATM features from epoch"""

    def __init__(self, threshold, duration):
        self.threshold = threshold
        self.duration = duration

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        nb_ROIs = np.shape(X)[1]
        nb_trials = np.shape(X)[0]

        temp = np.transpose(X, (1, 0, 2))
        temp_nc = np.reshape(temp, (np.shape(temp)[0], np.shape(temp)[1] * np.shape(temp)[2]))
        zscored_data = zscore(temp_nc, axis=1)
        # epoching here before computing the avalanches
        temp_zscored_data_ep = np.reshape(zscored_data, (np.shape(temp)[0], np.shape(temp)[1], np.shape(temp)[2]))
        zscored_data_ep = np.transpose(temp_zscored_data_ep, (1, 0, 2))

        ATM = np.empty((nb_trials, nb_ROIs, nb_ROIs))
        matrix_rd=np.ones((nb_ROIs,nb_ROIs))
        mask = np.tril(matrix_rd)
        for kk_trial in range(nb_trials):
            list_avalanches, min_size_avalanches, max_size_avalanches, list_avalanches_bysize, temp_ATM = find_avalanches(
                zscored_data_ep[kk_trial, :, :], thresh=self.threshold, val_duration=self.duration )
            # ATM: nb_trials x nb_ROIs x nb_ROIs matrix
            # to keep the half of the symmetrical matrix:
            ATM[kk_trial, :] = temp_ATM[mask==1] # already a vector
            # # reshape for SVM
            # reshape_ATM = np.reshape(ATM, (np.shape(ATM)[0], np.shape(ATM)[1] * np.shape(ATM)[2]))

        return ATM
