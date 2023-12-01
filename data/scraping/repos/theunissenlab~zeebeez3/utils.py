import os
import sys
from copy import deepcopy

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from soundsig.signal import coherency
from soundsig.spikes import compute_psth
from soundsig.timefreq import power_spectrum_jn
from zeebeez3.transforms.biosound import BiosoundTransform
from zeebeez3.transforms.pairwise_cf import PairwiseCFTransform
from zeebeez3.transforms.stim_event import StimEventTransform
from zeebeez3.core.utils import USED_ACOUSTIC_PROPS, ROSTRAL_CAUDAL_ELECTRODES_LEFT, ROSTRAL_CAUDAL_ELECTRODES_RIGHT

COLOR_BLUE_LFP = '#0068A5'
COLOR_YELLOW_SPIKE = '#F0DB00'
COLOR_RED_SPIKE_RATE = '#E90027'
COLOR_PURPLE_LFP_CROSS = '#863198'
COLOR_CRIMSON_SPIKE_SYNC = '#610B0B'


def set_font(size=16):
    font = {'family': 'normal', 'weight': 'bold', 'size': size}
    matplotlib.rc('font', **font)
    matplotlib.rc('axes', labelsize=24)
    matplotlib.rc('axes', titleweight='bold')


def get_this_dir():
    """ Get the directory that contains the python file that is calling this function. """

    f = sys._current_frames().values()[0]
    calling_file_path = f.f_back.f_globals['__file__']
    root_dir, fname = os.path.split(calling_file_path)
    return root_dir


def get_freqs(sample_rate, window_length=0.060, increment=None):
    if increment is None:
        increment = 2.0 / sample_rate
    nt = int(window_length * 2 * sample_rate)
    s = np.random.randn(nt)
    pfreq, psd1, ps_var, phase = power_spectrum_jn(s, sample_rate, window_length, increment)
    return pfreq


def get_lags_ms(sample_rate, lags=np.arange(-20, 21, 1)):
    return (lags / sample_rate) * 1e3


def log_transform(s):
    nz = s > 0
    s[nz] = 20 * np.log10(s[nz]) + 70
    s[s < 0] = 0


def compute_spectra_and_coherence_single_electrode(lfp1, lfp2, sample_rate, e1, e2,
                                                   window_length=0.060, increment=None, log=True,
                                                   window_fraction=0.60, noise_floor_db=25,
                                                   lags=np.arange(-20, 21, 1), psd_stats=None):
    """

    :param lfp1: An array of shape (ntrials, nt)
    :param lfp2: An array of shape (ntrials, nt)
    :return:
    """

    # compute the mean (locked) spectra
    lfp1_mean = lfp1.mean(axis=0)
    lfp2_mean = lfp2.mean(axis=0)

    if increment is None:
        increment = 2.0 / sample_rate

    pfreq, psd1, ps_var, phase = power_spectrum_jn(lfp1_mean, sample_rate, window_length, increment)
    pfreq, psd2, ps_var, phase = power_spectrum_jn(lfp2_mean, sample_rate, window_length, increment)

    if log:
        log_transform(psd1)
        log_transform(psd2)

    c12 = coherency(lfp1_mean, lfp2_mean, lags, window_fraction=window_fraction, noise_floor_db=noise_floor_db)

    # compute the nonlocked spectra coherence
    c12_pertrial = list()
    ntrials, nt = lfp1.shape
    psd1_ms_all = list()
    psd2_ms_all = list()
    for k in range(ntrials):
        i = np.ones([ntrials], dtype='bool')
        i[k] = False
        lfp1_jn_mean = lfp1[i, :].mean(axis=0)
        lfp2_jn_mean = lfp2[i, :].mean(axis=0)

        lfp1_ms = lfp1[k, :] - lfp1_jn_mean
        lfp2_ms = lfp2[k, :] - lfp2_jn_mean

        pfreq, psd1_ms, ps_var_ms, phase_ms = power_spectrum_jn(lfp1_ms, sample_rate, window_length, increment)
        pfreq, psd2_ms, ps_var_ms, phase_ms = power_spectrum_jn(lfp2_ms, sample_rate, window_length, increment)
        if log:
            log_transform(psd1_ms)
            log_transform(psd2_ms)

        psd1_ms_all.append(psd1_ms)
        psd2_ms_all.append(psd2_ms)

        c12_ms = coherency(lfp1_ms, lfp2_ms, lags, window_fraction=window_fraction, noise_floor_db=noise_floor_db)
        c12_pertrial.append(c12_ms)

    psd1_ms_all = np.array(psd1_ms_all)
    psd2_ms_all = np.array(psd2_ms_all)
    psd1_ms = psd1_ms_all.mean(axis=0)
    psd2_ms = psd2_ms_all.mean(axis=0)

    if psd_stats is not None:
        psd_mean1, psd_std1 = psd_stats[e1]
        psd_mean2, psd_std2 = psd_stats[e2]
        psd1 -= psd_mean1
        psd1 /= psd_std1
        psd2 -= psd_mean2
        psd2 /= psd_std2

        psd1_ms -= psd_mean1
        psd1_ms /= psd_std1
        psd2_ms -= psd_mean2
        psd2_ms /= psd_std2

    c12_pertrial = np.array(c12_pertrial)
    c12_nonlocked = c12_pertrial.mean(axis=0)

    # compute the coherence per trial then take the average
    c12_totals = list()
    for k in range(ntrials):
        c12 = coherency(lfp1[k, :], lfp2[k, :], lags, window_fraction=window_fraction, noise_floor_db=noise_floor_db)
        c12_totals.append(c12)

    c12_totals = np.array(c12_totals)
    c12_total = c12_totals.mean(axis=0)

    return pfreq, psd1, psd2, psd1_ms, psd2_ms, c12, c12_nonlocked, c12_total


def compute_spectra_and_coherence_multi_electrode_single_trial(lfps, sample_rate, electrode_indices, electrode_order,
                                                               window_length=0.060, increment=None, log=True,
                                                               window_fraction=0.60, noise_floor_db=25,
                                                               lags=np.arange(-20, 21, 1),
                                                               psd_stats=None):
    """
    :param lfps: an array of shape (ntrials, nelectrodes, nt)
    :return:
    """

    if increment is None:
        increment = 2.0 / sample_rate

    nelectrodes, nt = lfps.shape
    freqs = get_freqs(sample_rate, window_length, increment)
    lags_ms = get_lags_ms(sample_rate, lags)

    spectra = np.zeros([nelectrodes, len(freqs)])
    cross_mat = np.zeros([nelectrodes, nelectrodes, len(lags_ms)])

    for k in range(nelectrodes):

        _e1 = electrode_indices[k]
        i1 = electrode_order.index(_e1)

        lfp1 = lfps[k, :]

        freqs, psd1, ps_var, phase = power_spectrum_jn(lfp1, sample_rate, window_length, increment)
        if log:
            log_transform(psd1)

        if psd_stats is not None:
            psd_mean, psd_std = psd_stats[_e1]

            """
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.plot(freqs, psd1, 'k-')
            plt.title('PSD (%d)' % _e1)
            plt.axis('tight')

            plt.subplot(2, 2, 3)
            plt.plot(freqs, psd_mean, 'g-')
            plt.title('Mean')
            plt.axis('tight')

            plt.subplot(2, 2, 4)
            plt.plot(freqs, psd_std, 'c-')
            plt.title('STD')
            plt.axis('tight')

            plt.subplot(2, 2, 2)
            psd1_z = deepcopy(psd1)
            psd1_z -= psd_mean
            psd1_z /= psd_std
            plt.plot(freqs, psd1_z, 'r-')
            plt.title('Zscored')
            plt.axis('tight')
            """
            psd1 -= psd_mean
            psd1 /= psd_std

        spectra[i1, :] = psd1

        for j in range(k):
            _e2 = electrode_indices[j]
            i2 = electrode_order.index(_e2)

            lfp2 = lfps[j, :]

            cf = coherency(lfp1, lfp2, lags, window_fraction=window_fraction, noise_floor_db=noise_floor_db)

            """
            freqs,c12,c_var_amp,c_phase,c_phase_var,coherency,coherency_t = coherence_jn(lfp1, lfp2, sample_rate,
                                                                                         window_length, increment,
                                                                                         return_coherency=True)
            """

            cross_mat[i1, i2] = cf
            cross_mat[i2, i1] = cf[::-1]

    return spectra, cross_mat


def add_region_info(agg, df):
    """ Make a new DataFrame that contains region information. """

    edf = pd.read_csv('/auto/tdrive/mschachter/data/aggregate/electrode_data.csv')

    new_data = dict()
    for key in df.keys():
        new_data[key] = list()

    # peek into the aggregate data to get a list of class names
    k1 = agg.class_names.keys()[0]
    stim_class_names = agg.class_names[k1][0]

    new_data['reg1'] = list()
    new_data['reg2'] = list()
    new_data['gs'] = list()  # global selectivity

    for cname in stim_class_names:
        new_data['pcc_%s' % cname] = list()
        new_data['sel_%s' % cname] = list()

    # make a map of bird/block/hemi/electrode for fast lookup
    emap = dict()
    for k, row in edf.iterrows():
        key = (row['bird'], row['block'], row['hemisphere'], row['electrode'])
        reg = row['region']
        emap[key] = reg

    for k, row in df.iterrows():
        for key in df.keys():
            if key == 'segment' and row[key] == 'Call1c':
                new_data[key].append('Call1')
            else:
                new_data[key].append(row[key])

        bird = row['bird']
        block = row['block']
        hemi = row['hemi']
        e1 = row['e1']
        e2 = row['e2']

        # get the confusion matrix for this row
        index = row['index']
        mat_key = (row['decomp'], row['order'], row['ptype'])
        C = agg.confidence_matrices[mat_key][index]
        cnames = agg.class_names[mat_key][index]

        # compute the pcc fraction for each category
        pcc_fracs = np.zeros([len(cnames)])
        for k, cname in enumerate(cnames):
            p = C[k]
            p /= p.sum()
            pcc_fracs[k] = p[k]
            new_data['pcc_%s' % cname].append(p[k])

        # compute the selectivity for each category
        for k, cname in enumerate(cnames):
            i = np.ones(len(cnames), dtype='bool')
            i[k] = False
            sel = np.log2(((len(cnames) - 1) * pcc_fracs[k]) / pcc_fracs[i].sum())
            new_data['sel_%s' % cname].append(sel)

        # normalize the fractions so they become a distribution
        pcc_fracs /= pcc_fracs.sum()

        # compute the global selectivity
        if np.isnan(pcc_fracs).sum() > 0:
            gs = 0
        else:
            nz = pcc_fracs > 0
            assert np.abs(pcc_fracs.sum() - 1) < 1e-6, "pcc_fracs.sum()=%f" % pcc_fracs.sum()
            Hobs = -np.sum(pcc_fracs[nz] * np.log2(pcc_fracs[nz]))
            Hmax = np.log2(len(cnames))
            gs = 1. - (Hobs / Hmax)
        new_data['gs'].append(gs)

        key = (bird, block, hemi, e1)
        reg1 = emap[key]

        key = (bird, block, hemi, e2)
        reg2 = emap[key]

        reg1 = reg1.replace('L2b', 'L2')
        reg1 = reg1.replace('L2A', 'L2')
        reg1 = reg1.replace('L2B', 'L2')

        reg2 = reg2.replace('L2b', 'L2')
        reg2 = reg2.replace('L2A', 'L2')
        reg2 = reg2.replace('L2B', 'L2')

        new_data['reg1'].append(reg1)
        new_data['reg2'].append(reg2)

    return pd.DataFrame(new_data), stim_class_names


def get_psd_stats(bird, block, seg, hemi, data_dir='/auto/tdrive/mschachter/data'):
    transforms_dir = os.path.join(data_dir, bird, 'transforms')
    cf_file = os.path.join(transforms_dir, 'PairwiseCF_%s_%s_%s_%s_raw.h5' % (bird, block, seg, hemi))
    cft = PairwiseCFTransform.load(cf_file)

    electrodes = cft.df.electrode1.unique()

    estats = dict()
    for e in electrodes:
        i = (cft.df.electrode1 == e) & (cft.df.electrode1 == cft.df.electrode2) & (cft.df.decomp == 'locked')
        indices = cft.df['index'][i].values
        psds = cft.psds[indices]
        log_transform(psds)
        estats[e] = (psds.mean(axis=0), psds.std(axis=0, ddof=1))
    return estats


def compute_avg_and_ms(lfp):
    lfp_mean = lfp.mean(axis=0)

    lfp_ms_all = list()
    ntrials, nelectrodes = lfp.shape
    for k in range(ntrials):
        i = np.ones([ntrials], dtype='bool')
        i[k] = False
        lfp_resid = lfp[k, :] - lfp[i].mean(axis=0)
        lfp_ms_all.append(lfp_resid)

    lfp_ms_all = np.array(lfp_ms_all)
    lfp_ms = lfp_ms_all.mean(axis=0)

    return lfp_mean, lfp_ms


def get_e2e_dists(data_dir='/auto/tdrive/mschachter/data'):
    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data+dist.csv'))

    # precompute distance from each electrode to each other electrode
    e2e_dists = dict()
    for (bird, block, hemi), gdf in edata.groupby(['bird', 'block', 'hemisphere']):

        mult = 1.
        if bird == 'GreBlu9508M':
            mult = 4.

        num_electrodes = len(gdf.electrode.unique())
        assert num_electrodes == 16
        e2e = dict()
        for e1 in gdf.electrode.unique():
            i1 = (gdf.electrode == e1)
            assert i1.sum() == 1
            dl2a1 = gdf.dist_l2a[i1].values[0] * mult
            dmid1 = gdf.dist_midline[i1].values[0]

            for e2 in gdf.electrode.unique():
                i2 = (gdf.electrode == e2)
                assert i2.sum() == 1
                dl2a2 = gdf.dist_l2a[i2].values[0] * mult
                dmid2 = gdf.dist_midline[i2].values[0]

                e2e[(e1, e2)] = np.sqrt((dl2a1 - dl2a2) ** 2 + (dmid1 - dmid2) ** 2)
        e2e_dists[(bird, block, hemi)] = e2e

    return e2e_dists


def get_full_data(bird, block, segment, hemi, stim_id, data_dir='/auto/tdrive/mschachter/data'):
    bdir = os.path.join(data_dir, bird)
    tdir = os.path.join(bdir, 'transforms')

    aprops = USED_ACOUSTIC_PROPS

    # load the BioSound
    bs_file = os.path.join(tdir, 'BiosoundTransform_%s.h5' % bird)
    bs = BiosoundTransform.load(bs_file)

    # load the StimEvent transform
    se_file = os.path.join(tdir, 'StimEvent_%s_%s_%s_%s.h5' % (bird, block, segment, hemi))
    print
    'Loading %s...' % se_file
    se = StimEventTransform.load(se_file, rep_types_to_load=['raw'])
    se.zscore('raw')
    se.segment_stims_from_biosound(bs_file)

    # load the pairwise CF transform
    pcf_file = os.path.join(tdir, 'PairwiseCF_%s_%s_%s_%s_raw.h5' % (bird, block, segment, hemi))
    print
    'Loading %s...' % pcf_file
    pcf = PairwiseCFTransform.load(pcf_file)

    def log_transform(x, dbnoise=100.):
        x /= x.max()
        zi = x > 0
        x[zi] = 20 * np.log10(x[zi]) + dbnoise
        x[x < 0] = 0
        x /= x.max()

    all_lfp_psds = deepcopy(pcf.psds)
    log_transform(all_lfp_psds)
    all_lfp_psds -= all_lfp_psds.mean(axis=0)
    all_lfp_psds /= all_lfp_psds.std(axis=0, ddof=1)

    # get overall biosound stats
    bs_stats = dict()
    for aprop in aprops:
        amean = bs.stim_df[aprop].mean()
        astd = bs.stim_df[aprop].std(ddof=1)
        bs_stats[aprop] = (amean, astd)

    for (stim_id2, stim_type2), gdf in se.segment_df.groupby(['stim_id', 'stim_type']):
        print
        '%d: %s' % (stim_id2, stim_type2)

    # get the spectrogram
    i = se.segment_df.stim_id == stim_id
    last_end_time = se.segment_df.end_time[i].max()

    spec_freq = se.spec_freq
    stim_spec = se.spec_by_stim[stim_id]
    spec_t = np.arange(stim_spec.shape[1]) / se.lfp_sample_rate
    speci = np.min(np.where(spec_t > last_end_time)[0])
    spec_t = spec_t[:speci]
    stim_spec = stim_spec[:, :speci]
    stim_dur = spec_t.max() - spec_t.min()

    # get the raw LFP
    si = int(se.pre_stim_time * se.lfp_sample_rate)
    ei = int(stim_dur * se.lfp_sample_rate) + si
    lfp = se.lfp_reps_by_stim['raw'][stim_id][:, :, si:ei]
    ntrials, nelectrodes, nt = lfp.shape

    # get the raw spikes, spike_mat is ragged array of shape (num_trials, num_cells, num_spikes)
    spike_mat = se.spikes_by_stim[stim_id]
    assert ntrials == len(spike_mat)

    ncells = len(se.cell_df)
    print
    'ncells=%d' % ncells
    ntrials = len(spike_mat)

    # compute the PSTH
    psth = list()
    for n in range(ncells):
        # get the spikes across all trials for neuron n
        spikes = [spike_mat[k][n] for k in range(ntrials)]
        # make a PSTH
        _psth_t, _psth = compute_psth(spikes, stim_dur, bin_size=1.0 / se.lfp_sample_rate)
        psth.append(_psth)
    psth = np.array(psth)

    if hemi == 'L':
        electrode_order = ROSTRAL_CAUDAL_ELECTRODES_LEFT
    else:
        electrode_order = ROSTRAL_CAUDAL_ELECTRODES_RIGHT

    # get acoustic props and LFP/spike power spectra for each syllable
    syllable_props = list()

    i = bs.stim_df.stim_id == stim_id
    orders = sorted(bs.stim_df.order[i].values)
    cell_index2electrode = None
    for o in orders:
        i = (bs.stim_df.stim_id == stim_id) & (bs.stim_df.order == o)
        assert i.sum() == 1

        d = dict()
        d['start_time'] = bs.stim_df.start_time[i].values[0]
        d['end_time'] = bs.stim_df.end_time[i].values[0]
        d['order'] = o

        for aprop in aprops:
            amean, astd = bs_stats[aprop]
            d[aprop] = (bs.stim_df[aprop][i].values[0] - amean) / astd

        # get the LFP power spectra
        lfp_psd = list()
        for k, e in enumerate(electrode_order):
            i = (pcf.df.stim_id == stim_id) & (pcf.df.order == o) & (pcf.df.decomp == 'full') & \
                (pcf.df.electrode1 == e) & (pcf.df.electrode2 == e)

            assert i.sum() == 1, "i.sum()=%d" % i.sum()

            index = pcf.df[i]['index'].values[0]
            lfp_psd.append(all_lfp_psds[index, :])
        d['lfp_psd'] = np.array(lfp_psd)

        syllable_props.append(d)

    return {'stim_id': stim_id, 'spec_t': spec_t, 'spec_freq': spec_freq, 'spec': stim_spec,
            'lfp': lfp, 'spikes': spike_mat, 'lfp_sample_rate': se.lfp_sample_rate, 'psth': psth,
            'syllable_props': syllable_props, 'electrode_order': electrode_order, 'psd_freq': pcf.freqs,
            'cell_index2electrode': cell_index2electrode, 'aprops': aprops}


def get_electrode_dict(data_dir='/auto/tdrive/mschachter/data'):
    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data+dist.csv'))

    edict = dict()

    g = edata.groupby(['bird', 'block', 'hemisphere', 'electrode'])

    for (bird, block, hemi, electrode), gdf in g:
        assert len(gdf) == 1

        reg = clean_region(gdf.region.values[0])
        dist_l2a = gdf.dist_l2a.values[0]
        dist_midline = gdf.dist_midline.values[0]

        if bird == 'GreBlu9508M':
            dist_l2a *= 4

        edict[(bird, block, hemi, electrode)] = {'region': reg, 'dist_l2a': dist_l2a, 'dist_midline': dist_midline}

    return edict
