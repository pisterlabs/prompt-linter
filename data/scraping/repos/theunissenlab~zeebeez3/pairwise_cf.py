from copy import deepcopy
import os
import h5py

import numpy as np
from sklearn.decomposition import PCA

from soundsig.plots import multi_plot
from soundsig.sound import plot_spectrogram

from soundsig.spikes import plot_raster, simple_synchrony
import pandas as pd

import matplotlib.pyplot as plt

from soundsig.signal import coherency
from soundsig.timefreq import power_spectrum_jn

from zeebeez3.transforms.stim_event import StimEventTransform
from zeebeez3.core.utils import USED_ACOUSTIC_PROPS, ACOUSTIC_FUND_PROPS, decode_if_bytes, decode_column_if_bytes
from zeebeez3.aggregators.biosound import AggregateBiosounds


class PairwiseCFTransform(object):

    def __init__(self):
        self.data = None
        self.df = None

        self.psds = None
        self.cross_cfs = None

        self.spike_rate = None
        self.spike_synchrony = None

        self.freqs = None
        self.lags = None
        self.rep_type = None

        self.bird = None
        self.segment_uname = None
        self.rcg_names = None
        self.cell_index2electrode = None

    def transform(self, stim_event, lags=np.arange(-10, 11, 1), min_syllable_dur=0.050, post_syllable_dur=0.030,
                  rep_type='raw', debug=False, window_fraction = 0.60, noise_db = 25.):

        assert isinstance(stim_event, StimEventTransform)
        assert rep_type in stim_event.lfp_reps_by_stim
        self.rep_type = rep_type
        self.bird = stim_event.bird

        self.segment_uname = stim_event.seg_uname
        self.rcg_names = stim_event.rcg_names

        self.lags = (lags / stim_event.lfp_sample_rate)*1e3
        stim_ids = list(stim_event.lfp_reps_by_stim[rep_type].keys())

        all_psds = list()
        all_cross_cfs = list()

        all_spike_rates = list()
        all_spike_sync = list()

        # zscore the LFPs
        stim_event.zscore(rep_type)

        data = {'stim_id':list(), 'stim_type':list(), 'stim_duration':list(), 'order':list(), 'decomp':list(),
                'electrode1':list(), 'electrode2':list(), 'region1':list(), 'region2':list(),
                'cell_index':list(), 'cell_index2':list(), 'index':list()}

        # get map of electrode indices to region
        index2region = list()
        for e in stim_event.index2electrode:
            i = stim_event.electrode_data['electrode'] == e
            index2region.append(stim_event.electrode_data['region'][i][0])
        print('index2region=',index2region)

        # map cell indices to electrodes
        ncells = len(stim_event.cell_df)
        if ncells > 0:
            print('ncells=%d' % ncells)
            cell_index2electrode = [0]*ncells
            assert len(stim_event.cell_df.sort_code.unique()) == 1
            for ci,e in zip(stim_event.cell_df['index'], stim_event.cell_df['electrode']):
                cell_index2electrode[ci] = e
        else:
            cell_index2electrode = [-1]

        print('len(cell_index2electrode)=%d' % len(cell_index2electrode))

        # make a list of all valid syllables for each stimulus
        num_valid_syllables_per_type = dict()
        stim_syllables = dict()
        lags_max = np.abs(lags).max() / stim_event.lfp_sample_rate
        good_stim_ids = list()
        for stim_id in stim_ids:
            seg_times = list()
            i = stim_event.segment_df['stim_id'] == stim_id
            if i.sum() == 0:
                print('Missing stim information for stim %d!' % stim_id)
                continue

            stype = stim_event.segment_df['stim_type'][i].values[0]
            for k,(stime,etime,order) in enumerate(zip(stim_event.segment_df['start_time'][i], stim_event.segment_df['end_time'][i], stim_event.segment_df['order'][i])):
                dur = (etime - stime) + post_syllable_dur
                if dur < min_syllable_dur:
                    continue

                # make sure the duration is long enough to support the lags
                assert dur > lags_max, "Lags is too long, duration=%0.3f, lags_max=%0.3f" % (dur, lags_max)

                # add the syllable to the list, add in the extra post-stimulus time
                seg_times.append( (stime, etime+post_syllable_dur, order))
                if stype not in num_valid_syllables_per_type:
                    num_valid_syllables_per_type[stype] = 0
                num_valid_syllables_per_type[stype] += 1

            if len(seg_times) > 0:
                stim_syllables[stim_id] = np.array(seg_times)
                good_stim_ids.append(stim_id)

        print('# of syllables per category:')
        for stype,nstype in list(num_valid_syllables_per_type.items()):
            print('%s: %d' % (stype, nstype))

        # specify the window size for the auto-spectra and cross-coherence. the minimum segment size is 80ms
        psd_window_size = 0.060
        psd_increment = 2 / stim_event.lfp_sample_rate

        for stim_id in good_stim_ids:
            # get stim type
            i = stim_event.trial_df['stim_id'] == stim_id
            stim_type = stim_event.trial_df['stim_type'][i].values[0]

            print('Computing CFs for stim %d (%s)' % (stim_id, stim_type))

            # get the raw LFP
            X = stim_event.lfp_reps_by_stim[rep_type][stim_id]
            ntrials,nelectrodes,nt = X.shape

            # get the spike trains, a ragged array of shape (num_trials, num_cells, num_spikes)
            # The important thing to know about the spike times is that they are with respect
            # to the stimulus onset. Negative spike times occur prior to stimulus onset.
            spike_mat = stim_event.spikes_by_stim[stim_id]
            assert ntrials == len(spike_mat), "Weird number of trials in spike_mat: %d" % (len(spike_mat))

            # get segment data for this stim, start and end times of segments
            seg_times = stim_syllables[stim_id]

            # go through each syllable of the stimulus
            for stime,etime,order in seg_times:
                # compute the start and end indices of the LFP for this syllable, keeping in mind that the LFP
                # segment for this stimulus includes the pre and post stim times.
                lfp_syllable_start = (stim_event.pre_stim_time + stime)
                lfp_syllable_end = (stim_event.pre_stim_time + etime)
                si = int(lfp_syllable_start*stim_event.lfp_sample_rate)
                ei = int(lfp_syllable_end*stim_event.lfp_sample_rate)

                stim_dur = etime - stime

                # because spike times are with respect to the onset of the first syllable
                # of this stimulus, stime and etime define the appropriate window of analysis
                # for this syllable for spike times.
                spike_syllable_start = stime
                spike_syllable_end = etime
                spike_syllable_dur = etime - stime

                if debug:
                    # get the spectrogram, lfp and spikes for the stimulus
                    stim_spec = stim_event.spec_by_stim[stim_id]
                    syllable_spec = stim_spec[:, si:ei]
                    the_lfp = X[:, :, si:ei]
                    the_spikes = list()
                    lfp_t = np.arange(the_lfp.shape[-1]) / stim_event.lfp_sample_rate
                    spec_t = np.arange(syllable_spec.shape[1]) / stim_event.lfp_sample_rate
                    spec_freq = stim_event.spec_freq

                    for k in range(ntrials):
                        the_cell_spikes = list()
                        for n in range(ncells):
                            st = spike_mat[k][n]
                            i = (st >= stime) & (st <= etime)
                            print('trial=%d, cell=%d, nspikes=%d, (%0.3f, %0.3f)' % (k, n, i.sum(), spike_syllable_start, spike_syllable_end))
                            print('st=',st)
                            the_cell_spikes.append(st[i] - stime)
                        the_spikes.append(the_cell_spikes)

                    # make some plots to check the raw data, left hand plot is LFP, right hand is spikes
                    nrows = ntrials + 1
                    plt.figure()
                    gs = plt.GridSpec(nrows, 2)

                    ax = plt.subplot(gs[0, 0])
                    plot_spectrogram(spec_t, spec_freq, syllable_spec, ax=ax, colormap='gray', colorbar=False)

                    ax = plt.subplot(gs[0, 1])
                    plot_spectrogram(spec_t, spec_freq, syllable_spec, ax=ax, colormap='gray', colorbar=False)

                    for k in range(ntrials):
                        ax = plt.subplot(gs[k+1, 0])
                        lllfp = the_lfp[k, :, :]
                        absmax = np.abs(lllfp).max()
                        plt.imshow(lllfp, interpolation='nearest', aspect='auto', cmap=plt.cm.seismic,
                                   vmin=-absmax, vmax=absmax, origin='lower', extent=[lfp_t.min(), lfp_t.max(), 1, nelectrodes])

                        ax = plt.subplot(gs[k+1, 1])
                        plot_raster(the_spikes[k], ax=ax, duration=spike_syllable_end-spike_syllable_start, time_offset=0,
                                    ylabel='')

                    plt.title('stim %d, syllable %d' % (stim_id, order))
                    plt.show()

                # compute the LFP props
                lfp_props = self.compute_lfp_spectra_and_cfs(X[:, :, si:ei], stim_event.lfp_sample_rate,
                                                             lags, psd_window_size, psd_increment, window_fraction,
                                                             noise_db)

                # save the power spectra to the data frame and data matrix
                decomp_types = ['trial_avg', 'mean_sub', 'full', 'onewin']
                for n,e in enumerate(stim_event.index2electrode):
                    for decomp in decomp_types:

                        data['stim_id'].append(stim_id)
                        data['stim_type'].append(stim_type)
                        data['order'].append(order)
                        data['decomp'].append(decomp)
                        data['electrode1'].append(e)
                        data['electrode2'].append(e)
                        data['region1'].append(index2region[n])
                        data['region2'].append(index2region[n])
                        data['cell_index'].append(-1)
                        data['cell_index2'].append(-1)
                        data['index'].append(len(all_psds))
                        data['stim_duration'].append(stim_dur)

                        pkey = '%s_psds' % decomp
                        all_psds.append(lfp_props[pkey][n])

                # save the cross terms to the data frame and data matrix
                for n1,e1 in enumerate(stim_event.index2electrode):
                    for n2 in range(n1):
                        e2 = stim_event.index2electrode[n2]

                        # get index of cfs for this electrode pair
                        pi = lfp_props['cross_electrodes'].index( (n1, n2) )
                        for decomp in decomp_types:

                            if decomp == 'onewin':
                                continue

                            data['stim_id'].append(stim_id)
                            data['stim_type'].append(stim_type)
                            data['order'].append(order)
                            data['decomp'].append(decomp)
                            data['electrode1'].append(e1)
                            data['electrode2'].append(e2)
                            data['region1'].append(index2region[n1])
                            data['region2'].append(index2region[n2])
                            data['cell_index'].append(-1)
                            data['cell_index2'].append(-1)
                            data['index'].append(len(all_cross_cfs))
                            data['stim_duration'].append(stim_dur)

                            pkey = '%s_cfs' % decomp
                            all_cross_cfs.append(lfp_props[pkey][pi])

                if len(cell_index2electrode) == 1 and cell_index2electrode[0] == -1:
                    continue

                # compute the spike rate vector for each neuron
                for ci,e in enumerate(cell_index2electrode):
                    rates = list()
                    for k in range(ntrials):
                        st = spike_mat[k][ci]

                        i = (st >= spike_syllable_start) & (st <= spike_syllable_end)
                        r = i.sum() / (spike_syllable_end - spike_syllable_start)
                        rates.append(r)

                    rates = np.array(rates)
                    rv = [rates.mean(), rates.std(ddof=1)]

                    data['stim_id'].append(stim_id)
                    data['stim_type'].append(stim_type)
                    data['order'].append(order)
                    data['decomp'].append('spike_rate')
                    data['electrode1'].append(e)
                    data['electrode2'].append(e)
                    data['region1'].append(index2region[n])
                    data['region2'].append(index2region[n])
                    data['cell_index'].append(ci)
                    data['cell_index2'].append(ci)
                    data['index'].append(len(all_spike_rates))
                    data['stim_duration'].append(stim_dur)

                    all_spike_rates.append(rv)

                # compute the pairwise synchrony between spike trains
                spike_sync = np.zeros([ntrials, ncells, ncells])
                for k in range(ntrials):
                    for ci1,e1 in enumerate(cell_index2electrode):
                        st1 = spike_mat[k][ci1]
                        if len(st1) == 0:
                            continue

                        for ci2 in range(ci1):
                            st2 = spike_mat[k][ci2]
                            if len(st2) == 0:
                                continue
                            sync12 = simple_synchrony(st1, st2, spike_syllable_dur, bin_size=3e-3)
                            spike_sync[k, ci1, ci2] = sync12
                            spike_sync[k, ci2, ci1] = sync12

                # average pairwise synchrony across trials
                spike_sync_avg = spike_sync.mean(axis=0)

                # save the spike sync data into the data frame
                for ci1, e1 in enumerate(cell_index2electrode):
                    n1 = stim_event.index2electrode.index(e1)
                    for ci2 in range(ci1):
                        e2 = cell_index2electrode[ci2]
                        n2 = stim_event.index2electrode.index(e2)

                        data['stim_id'].append(stim_id)
                        data['stim_type'].append(stim_type)
                        data['order'].append(order)
                        data['decomp'].append('spike_sync')
                        data['electrode1'].append(e1)
                        data['electrode2'].append(e2)
                        data['region1'].append(index2region[n1])
                        data['region2'].append(index2region[n2])
                        data['cell_index'].append(ci1)
                        data['cell_index2'].append(ci2)
                        data['index'].append(len(all_spike_sync))
                        data['stim_duration'].append(stim_dur)

                        all_spike_sync.append(spike_sync_avg[ci1, ci2])

        self.cell_index2electrode = cell_index2electrode

        self.psds = np.array(all_psds)
        self.cross_cfs = np.array(all_cross_cfs)

        self.spike_rate = np.array(all_spike_rates)
        self.spike_synchrony = np.array(all_spike_sync)

        self.data = data
        self.df = pd.DataFrame(self.data)

    def compute_lfp_spectra_and_cfs(self, lfp, sample_rate, lags, psd_window_size, psd_increment,
                                    window_fraction, noise_db):
        """ Compute the power spectrums and cross coherencies of the multi-electrode LFP.

        :param lfp: A matrix of LFPs in the shape (ntrials, nelectrodes, ntime)
        :param sample_rate: Sample rate of the LFP in Hz
        :param lags: Integer-valued time lags for computing cross coherencies
        :param psd_window_size: Window size in seconds used for computing the power spectra.
        :param psd_increment: Increment in seconds used for computing the power spectra.

        :return:
        """

        ntrials,nelectrodes,nt = lfp.shape

        # pre-compute the trial-averaged LFP
        trial_avg_lfp = np.zeros([nelectrodes, nt])
        for n in range(nelectrodes):
            trial_avg_lfp[n, :] = lfp[:, n, :].mean(axis=0)

        # pre-compute the mean-subtracted LFP
        mean_sub_lfp = np.zeros_like(lfp)
        for n in range(nelectrodes):
            for k in range(ntrials):
                i = np.ones([ntrials], dtype='bool')
                i[k] = False
                lfp_jn_mean = lfp[i, n, :].mean(axis=0)
                mean_sub_lfp[k, n, :] = lfp[k, n, :] - lfp_jn_mean

        # compute the power spectra and covariance functions three different ways
        trial_avg_psds = list()
        mean_sub_psds = list()
        full_psds = list()
        onewin_psds = list()
        for n in range(nelectrodes):

            # compute the PSD of the trial-averaged LFP
            freq1, trial_avg_psd, trial_avg_phase = self.compute_psd(trial_avg_lfp[n, :], sample_rate, psd_window_size, psd_increment)
            trial_avg_psds.append(trial_avg_psd)
            
            if self.freqs is None:
                self.freqs = freq1

            # compute the trial-averaged PSD of mean-subtracted LFP
            per_trial_psds = list()
            for k in range(ntrials):
                freq1, mean_sub_psd, mean_sub_phase = self.compute_psd(mean_sub_lfp[k, n, :], sample_rate, psd_window_size, psd_increment)
                per_trial_psds.append(mean_sub_psd)

            per_trial_psds = np.array(per_trial_psds)
            mean_sub_psds.append(per_trial_psds.mean(axis=0))

            # compute the "full" psd, per trial then averaged across trial
            per_trial_psds = list()
            for k in range(ntrials):
                freq1, trial_psd, trial_phase = self.compute_psd(lfp[k, n, :], sample_rate, psd_window_size, psd_increment)
                per_trial_psds.append(trial_psd)

            per_trial_psds = np.array(per_trial_psds)
            full_psds.append(per_trial_psds.mean(axis=0))

            # compute the "full" psd, but with only the first 65ms of the LFP, so there is one window
            per_trial_onewin_psds = list()
            onewin_len = psd_window_size + (2 / sample_rate)
            onewin_nlen = int(onewin_len*sample_rate)
            for k in range(ntrials):
                freq1, onewin_trial_psd, onewin_trial_phase = self.compute_psd(lfp[k, n, :onewin_nlen], sample_rate, psd_window_size, psd_increment)
                per_trial_onewin_psds.append(onewin_trial_psd)
            per_trial_onewin_psds = np.array(per_trial_onewin_psds)
            onewin_psds.append(per_trial_onewin_psds.mean(axis=0))

        # compute the cross coherencies three different ways
        cross_electrodes = list()
        trial_avg_cfs = list()
        mean_sub_cfs = list()
        full_cfs = list()
        for n1 in range(nelectrodes):
            for n2 in range(n1):
                cross_electrodes.append( (n1, n2) )

                # compute coherency of trial-averaged LFP
                trial_avg_cf = coherency(trial_avg_lfp[n1, :], trial_avg_lfp[n2, :], lags,
                                         window_fraction=window_fraction, noise_floor_db=noise_db)
                trial_avg_cfs.append(trial_avg_cf)

                # compute average mean-subtracted coherency
                trial_cfs = list()
                for k in range(ntrials):
                    cf = coherency(mean_sub_lfp[k, n1, :], mean_sub_lfp[k, n2, :], lags, window_fraction=window_fraction, noise_floor_db=noise_db)
                    trial_cfs.append(cf)
                trial_cfs = np.array(trial_cfs)
                mean_sub_cfs.append(trial_cfs.mean(axis=0))

                # compute full coherency for each trial, then average across trials
                trial_cfs = list()
                for k in range(ntrials):
                    cf = coherency(lfp[k, n1, :], lfp[k, n2, :], lags, window_fraction=window_fraction, noise_floor_db=noise_db)
                    trial_cfs.append(cf)
                trial_cfs = np.array(trial_cfs)
                full_cfs.append(trial_cfs.mean(axis=0))

        return {'trial_avg_psds':trial_avg_psds,
                'mean_sub_psds':mean_sub_psds,
                'full_psds':full_psds,
                'onewin_psds':onewin_psds,
                'trial_avg_cfs':trial_avg_cfs,
                'mean_sub_cfs':mean_sub_cfs,
                'full_cfs':full_cfs,
                'cross_electrodes':cross_electrodes}

    def compute_psd(self, s, sample_rate, window_length, increment):
        """ Computes the power spectrum of a signal.
        """

        min_freq = 0
        max_freq = sample_rate / 2

        freq,psd,psd_var,phase = power_spectrum_jn(s, sample_rate, window_length, increment, min_freq=min_freq, max_freq=max_freq)

        # zero out frequencies where the lower bound dips below zero
        pstd = np.sqrt(psd_var)
        psd_lb = psd - pstd
        psd[psd_lb < 0] = 0

        return freq,psd,phase

    def save(self, output_file):
        hf = h5py.File(output_file, 'w')
        col_names = list(self.data.keys())
        hf.attrs['bird'] = self.bird
        hf.attrs['rcg_names'] = self.rcg_names
        hf.attrs['segment_uname'] = self.segment_uname
        hf.attrs['rep_type'] = self.rep_type
        hf.attrs['col_names'] = col_names
        hf.attrs['lags'] = self.lags
        hf.attrs['freqs'] = self.freqs
        hf.attrs['cell_index2electrode'] = self.cell_index2electrode
        for cname in col_names:
            try:
                hf[cname] = np.array(self.data[cname])
            except Exception as e:
                print('exception, cname=%s' % cname)
                print(e)
                print(self.data[cname])

        hf['PSD'] = self.psds
        hf['XCF'] = self.cross_cfs

        if len(self.spike_rate) > 0:
            hf['SPIKE_RATE'] = self.spike_rate
            hf['SPIKE_SYNC'] = self.spike_synchrony
        hf.close()

    @classmethod
    def load(clz, cf_file):
        cft = PairwiseCFTransform()

        hf = h5py.File(cf_file, 'r')
        cft.rep_type = decode_if_bytes(hf.attrs['rep_type'])
        col_names = [decode_if_bytes(s) for s in hf.attrs['col_names']]
        cft.lags = hf.attrs['lags']
        cft.freqs = hf.attrs['freqs']
        cft.cell_index2electrode = hf.attrs['cell_index2electrode']

        cft.psds = np.array(hf['PSD'])
        cft.cross_cfs = np.array(hf['XCF'])

        if 'SPIKE_RATE' in list(hf.keys()):
            cft.spike_rate = np.array(hf['SPIKE_RATE'])
            cft.spike_synchrony = np.array(hf['SPIKE_SYNC'])

        cft.bird = decode_if_bytes(hf.attrs['bird'])
        cft.segment_uname = decode_if_bytes(hf.attrs['segment_uname'])
        cft.rcg_names = [decode_if_bytes(s) for s in hf.attrs['rcg_names']]
        cft.data = dict()
        for cname in col_names:
            cft.data[decode_if_bytes(cname)] = np.array(hf[cname])
        hf.close()
        cft.df = pd.DataFrame(cft.data)
        decode_column_if_bytes(cft.df)

        return cft

    def get_bs_data(self, bs_file, exclude_classes=('wnoise', 'mlnoise')):

        bst = AggregateBiosounds.load(bs_file)
        integer2prop = bst.acoustic_props

        i = bst.df.bird == self.bird
        df = bst.df[i]

        # estimate quantiles to do outlier detection later
        bs_quantiles = dict()
        for aprop in USED_ACOUSTIC_PROPS:
            i = bst.acoustic_props.index(aprop)
            q2 = np.percentile(bst.Xraw[:, i], 1)
            q98 = np.percentile(bst.Xraw[:, i], 99)
            bs_quantiles[aprop] = (q2, q98)

        # map the biosound data to a dictionary for fast lookup
        bs_data = dict()
        i = np.ones(len(df), dtype='bool')
        for etype in exclude_classes:
            i &= df['stim_type'] != etype
        g = df[i].groupby(['stim_id', 'syllable_order'])
        for (stim_id,order),gdf in g:
            assert len(gdf) == 1, "More than one entry in biosound df for stim id %d, order %d" % (stim_id, order)
            xindex = gdf.xindex.values[0]
            duration = gdf.end_time.values[0] - gdf.start_time.values[0]
            v = bst.Xraw[xindex, :]
            v = [vv for k,vv in enumerate(v) if bst.acoustic_props[k] in USED_ACOUSTIC_PROPS]
            bs_data[(stim_id, order)] = (v, duration)

        return bs_data, USED_ACOUSTIC_PROPS, bs_quantiles

    def export_for_acoustic_decoder(self, output_file,
                                    decomps=('full_psds', 'full_cfs'),
                                    exclude_classes=('wnoise', 'mlnoise'),
                                    bs_file=None,
                                    min_duration=0.040, max_duration=0.400,
                                    plot=False,
                                    merge_freqs=False):
        """ Export data in a format appropriate for the CategoricalDecoder or AcousticDecoder (in
            zeebeez.models.{categorical,acoustic}_decoder.py). """

        bs_data = dict()
        integer2prop = []
        if bs_file is not None:
            bs_data,integer2prop,bs_quantiles = self.get_bs_data(bs_file, exclude_classes=exclude_classes)

        # get the name of the bird
        sstrs = self.segment_uname.split('_')
        bird = sstrs[0]

        # read the aggregate stim file to get information about bird that emitted each call
        stim_df = pd.read_csv('/auto/tdrive/mschachter/data/aggregate/stim_data.csv')
        i = stim_df.bird == bird
        index2bird = list(stim_df[i].emitter.unique())

        # map the stim ids to emitters
        stimid2emitter = dict()
        for stim_id in self.df.stim_id.unique():
            i = (stim_df['id'] == stim_id) & (stim_df['bird'] == bird)
            assert i.sum() == 1, "More than one entry in stim_df for stim_id=%d, i.sum()=%d" % (stim_id, i.sum())
            e = stim_df[i]['emitter'].values[0]
            stimid2emitter[stim_id] = e

        # map the stim index and type to integers
        index2id = list()
        index2type = list()
        g = self.df.groupby(['stim_id', 'stim_type', 'order'])
        for (stim_id,stim_type,order),gdf in g:
            if stim_type in exclude_classes:
                continue

            sname = '%d_%d' % (stim_id, order)
            index2id.append(sname)
            if stim_type not in index2type:
                index2type.append(stim_type)

        index2electrode = self.df.electrode1.unique()

        X = list()
        S = list()
        Y = list()

        # normalize all power spectra and cross-coherencies with respect to eachother
        psds = None
        index2index = None
        freqs = self.freqs
        if decomps[0].endswith('psds'):
            dc = decomps[0].split('_')[0]
            i = (self.df.decomp == dc) & (self.df.electrode1 == self.df.electrode2)
            xi = self.df['index'][i].values
            index2index = {xii:k for k,xii in enumerate(xi)}

            # take log transform of normalized power spectra
            psds = deepcopy(self.psds[xi, :])
            psds /= psds.max()
            self.log_transform(psds)

            if merge_freqs:
                psds,freqs = self.merge_frequency_bands(psds)

        # get the log spike rates
        spike_rates = deepcopy(self.spike_rate)

        # get spike synchronies
        spike_sync = deepcopy(self.spike_synchrony)

        # go through each stimulus, and aggregate either correlation functions or spectra/coherences
        g = self.df.groupby(['stim_id', 'stim_type', 'order'])
        the_index = 0

        unique_decomps = self.df.decomp.unique()

        for (stim_id,stim_type,order),gdf in g:

            if stim_type in exclude_classes:
                continue

            if bs_file is not None:
                assert (stim_id,order) in bs_data, "Missing biosound stim: stim_id=%d, order=%d" % (stim_id, order)
                v_biosound,duration = bs_data[(stim_id, order)]
                if duration < min_duration or duration > max_duration:
                    print('Stim (%d,%d) is above or below duration boundary, min_dur=%0.3f, max_dur=%0.3f, dur=%0.3f' % \
                          (stim_id, order, min_duration, max_duration, duration))
                    continue

            # build up a vector x comprised of one or more decompositions of the LFP and cell spike trains
            x = list()
            for decomp in decomps:

                if decomp.endswith('psds'):
                    decomp0 = '_'.join(decomp.split('_')[:-1])
                    assert decomp0 in unique_decomps, 'Unknown decomposition: %s, available types are %s' % (decomp0, unique_decomps)
                    for e in index2electrode:
                        i = (gdf.electrode1 == e) & (gdf.electrode2 == e) & (gdf.decomp == decomp0)
                        assert i.sum() == 1, "Zero or many results for self, e=%d, decomp0=%s, i.sum()=%d" % (e, decomp0, i.sum())

                        index = gdf[i]['index'].values[0]
                        if index not in index2index:
                            print('weird index: e=%d, decomp0=%s, index=%d' % (e, decomp0, index))
                        x.extend(psds[index2index[index], :])

                elif decomp.endswith('cfs'):
                    decomp0 = '_'.join(decomp.split('_')[:-1])
                    assert decomp0 in unique_decomps, 'Unknown decomposition: %s, available types are %s' % (decomp0, unique_decomps)
                    sub_df = gdf[(gdf.electrode1 != gdf.electrode2) & (gdf.decomp == decomp0)]
                    for k,e1 in enumerate(index2electrode):
                        for j in range(k):
                            e2 = index2electrode[j]
                            i = (sub_df.electrode1 == e1) & (sub_df.electrode2 == e2)
                            if i.sum() == 0:
                                i = (sub_df.electrode1 == e2) & (sub_df.electrode2 == e1)
                            assert i.sum() == 1, "Zero or many results: e1=%d, e2=%d, decomp0=%s, i.sum()=%d" % \
                                                 (e1, e2, decomp0, i.sum())

                            index = sub_df[i]['index'].values[0]
                            x.extend(self.cross_cfs[index, :])

                elif decomp == 'spike_rate':
                    for ci,e in enumerate(self.cell_index2electrode):
                        i = (gdf.cell_index == ci) & (gdf.decomp == decomp)
                        assert i.sum() == 1

                        xindex = gdf[i]['index'].values[0]
                        rate_mean, rate_std = spike_rates[xindex]
                        x.append(rate_mean)

                elif decomp == 'spike_sync':
                    sub_df = gdf[(gdf.cell_index != -1) & (gdf.cell_index != gdf.cell_index2) & (gdf.decomp == decomp)]
                    for ci1,e1 in enumerate(self.cell_index2electrode):
                        for ci2 in range(ci1):
                            i = (sub_df.cell_index == ci1) & (sub_df.cell_index2 == ci2)
                            assert i.sum() == 1, "stim_id=%d, order=%d, ci1=%d, ci2=%d, i.sum()=%d" % (stim_id, order, ci1, ci2, i.sum())
                            xindex = sub_df[i]['index'].values[0]
                            x.append(spike_sync[xindex])

            sname = '%d_%d' % (stim_id, order)

            emitter = stimid2emitter[stim_id]

            X.append(x)
            Y.append( (index2type.index(stim_type), index2id.index(sname), 0, index2bird.index(emitter)))

            if bs_file is not None:
                S.append(np.array(v_biosound))
            else:
                S.append(the_index)

            the_index += 1

        X = np.array(X)
        S = np.array(S)
        Y = np.array(Y)

        # identify outliers in the dataset and remove them
        good_i = np.ones([S.shape[0]], dtype='bool')
        for k,aprop in enumerate(USED_ACOUSTIC_PROPS):
            good_i &= ~np.isnan(S[:, k])
            good_i &= ~np.isinf(S[:, k])
            if aprop not in ACOUSTIC_FUND_PROPS:
                good_i &= (S[:, k] >= bs_quantiles[aprop][0]) & (S[:, k] <= bs_quantiles[aprop][1])

        print('Removing %d outlier points from dataset, %d remaining' % ((~good_i).sum(), good_i.sum()))

        X = X[good_i, :]
        Y = Y[good_i, :]
        S = S[good_i, :]

        if plot:
            plist = list()
            for k,aprop in enumerate(USED_ACOUSTIC_PROPS):
                plist.append({'aprop':aprop, 'vals':S[:, k]})
            def _plt_hist(_pdata, _ax):
                plt.hist(_pdata['vals'], bins=20, color='b', alpha=0.7)
                plt.title(_pdata['aprop'])
            multi_plot(plist, _plt_hist, nrows=4, ncols=5)
            plt.show()

        hf = h5py.File(output_file, 'w')
        hf['X'] = X
        hf['S'] = S
        hf['Y'] = Y
        hf.attrs['integer2type'] = index2type
        hf.attrs['integer2id'] = index2id
        hf.attrs['integer2bird'] = index2bird
        hf.attrs['integer2prop'] = integer2prop
        hf.attrs['freqs'] = freqs
        hf.attrs['lags'] = self.lags
        hf.attrs['cell_index2electrode'] = self.cell_index2electrode
        hf.attrs['index2electrode'] = index2electrode

        hf.close()

    def merge_frequency_bands(self, X, bands=[(0, 30), (30, 80), (80, 190)]):
        Xnew = np.zeros([X.shape[0], len(bands)])
        freqs = np.zeros(len(bands))
        for b,(low_freq,high_freq) in enumerate(bands):
            fi = (self.freqs >= low_freq) & (self.freqs < high_freq)
            Xnew[:, b] = X[:, fi].sum(axis=1)
            freqs[b] = ((high_freq - low_freq) / 2.) + low_freq

        return Xnew,freqs

    def export_for_lfp_decoder(self, output_file, decomp='full', merge_freqs=True):

        sstrs = self.segment_uname.split('_')
        bird = sstrs[0]

        # read the aggregate stim file to get information about bird that emitted each call
        stim_df = pd.read_csv('/auto/tdrive/mschachter/data/aggregate/stim_data.csv')
        i = stim_df.bird == bird
        index2bird = list(stim_df[i].emitter.unique())

        # map the stim ids to emitters
        stimid2emitter = dict()
        for stim_id in self.df.stim_id.unique():
            i = (stim_df['id'] == stim_id) & (stim_df['bird'] == bird)
            assert i.sum() == 1, "More than one entry in stim_df for stim_id=%d, i.sum()=%d" % (stim_id, i.sum())
            e = stim_df[i]['emitter'].values[0]
            stimid2emitter[stim_id] = e

        # map the stim index and type to integers
        index2id = list()
        index2type = list()
        g = self.df.groupby(['stim_id', 'stim_type', 'order'])
        for (stim_id, stim_type, order), gdf in g:
            if stim_type in ['mlnoise']:
                continue

            sname = '%d_%d' % (stim_id, order)
            index2id.append(sname)
            if stim_type not in index2type:
                index2type.append(stim_type)

        # get the map between electrodes and cells
        index2electrode = list(self.df.electrode1.unique())
        index2cell = list(self.df.cell_index.unique())
        index2cell.remove(-1)

        print('index2electrode=',index2electrode)
        print('index2cell=',index2cell)
        print('ncells=%d' % len(self.cell_index2electrode))

        i = (self.df.stim_type != 'mlnoise')
        df = self.df[i]
        g = df.groupby(['stim_id', 'order'])

        # vector of spike rates for each syllable
        Xrate = list()

        # matrix of synchronies for each syllable
        Xsync = list()

        # matrix of lfp power for each syllable
        Ylfp = list()

        # matrix of pairwise correlation functions for each syllable
        Ycf = list()

        # matrix of stimulus information
        Ystim = list()

        # log transform the power spectra
        i = (df.decomp == decomp) & (df.electrode1 == df.electrode2)
        assert i.sum() > 0
        xi = df['index'][i].values
        psds = deepcopy(self.psds[xi, :])
        index2index = {xii:k for k,xii in enumerate(xi)}
        self.log_transform(psds)
        freqs = self.freqs
        if merge_freqs:
            psds,freqs = self.merge_frequency_bands(psds)

        for (stim_id,order),gdf in g:

            stim_type = gdf.stim_type.values[0]
            emitter = stimid2emitter[stim_id]
            sname = '%d_%d' % (stim_id, order)

            # create spike rate vector for this syllable
            r = list()
            for ci,e in enumerate(self.cell_index2electrode):
                i = (gdf.decomp == 'spike_rate') & (gdf.cell_index == ci)
                assert i.sum() == 1
                ii = gdf['index'][i].values[0]
                r.append(self.spike_rate[ii, 0])
            Xrate.append(r)

            # create a vector of synchronies for this syllable
            s = list()
            for ci1,e1 in enumerate(self.cell_index2electrode):
                for ci2 in range(ci1):
                    i = (gdf.decomp == 'spike_sync') & (gdf.cell_index == ci1) & (gdf.cell_index2 == ci2)
                    assert i.sum() == 1
                    xindex = gdf['index'][i].values[0]
                    s.append(self.spike_synchrony[xindex])
            Xsync.append(s)

            # create a vector of lfp psds for this syllable
            x = list()
            for k,e in enumerate(index2electrode):
                i = (gdf.decomp == decomp) & (gdf.electrode1 == e) & (gdf.electrode2 == e)
                assert i.sum() == 1, "i.sum()=%d" % i.sum()
                ii = gdf['index'][i].values[0]
                x.extend(psds[index2index[ii], :])
            Ylfp.append(x)

            # create a vector of pairwise cfs for this syllable
            z = list()
            for k,e1 in enumerate(index2electrode):
                for j in range(k):
                    e2 = index2electrode[j]
                    i = (gdf.decomp == 'full') & (gdf.electrode1 == e1) & (gdf.electrode2 == e2)
                    if i.sum() == 0:
                        i = (gdf.decomp == 'full') & (gdf.electrode1 == e2) & (gdf.electrode2 == e1)
                    assert i.sum() == 1
                    ii = gdf['index'][i]
                    z.extend(self.cross_cfs[ii, :])
            Ycf.append(z)

            # append the stimulus information
            Ystim.append((index2type.index(stim_type), index2id.index(sname), 0, index2bird.index(emitter)))

        hf = h5py.File(output_file, 'w')

        hf['Xrate'] = np.array(Xrate)
        hf['Xsync'] = np.array(Xsync)
        hf['Ylfp'] = np.array(Ylfp)
        hf['Ycf'] = np.array(Ycf)
        hf['Ystim'] = np.array(Ystim)

        hf.attrs['freqs'] = freqs
        hf.attrs['lags'] = self.lags
        hf.attrs['index2cell'] = index2cell
        hf.attrs['index2electrode'] = index2electrode
        hf.attrs['cell_index2electrode'] = self.cell_index2electrode

        hf.attrs['integer2type'] = index2type
        hf.attrs['integer2id'] = index2id
        hf.attrs['integer2bird'] = index2bird

        hf.close()

    def reduce_coherency(self, cfunc, lags):
        """ Reduce the coherency function to just it's lag-zero peak and left and right sums. """

        li = lags < 0
        lsum = np.abs(cfunc[li]).sum()

        ri = lags > 0
        rsum = np.abs(cfunc[ri]).sum()

        ci = lags == 0
        csum = np.abs(cfunc[ci]).sum()

        return np.array([lsum,csum,rsum])

    def log_transform(self, x, dbnoise=100):
        x /= x.max()
        zi = x > 0
        x[zi] = 20*np.log10(x[zi]) + dbnoise
        x[x < 0] = 0
        x /= x.max()

def merge_preprocs(preproc_file1, preproc_file2, output_file):
    
    attr_names = ['freqs', 'integer2bird', 'integer2id', 'integer2prop',
                  'integer2type', 'lags']

    hf1 = h5py.File(preproc_file1, 'r')    
    attrs = dict()
    for attr in attr_names:
        attrs[attr] = hf1.attrs[attr]
        
    cell_index2electrode1 = hf1.attrs['cell_index2electrode']
    index2electrode1 = hf1.attrs['index2electrode']

    S1 = np.array(hf1['S'])
    X1 = np.array(hf1['X'])
    Y1 = np.array(hf1['Y'])

    hf1.close()

    hf2 = h5py.File(preproc_file2, 'r')

    cell_index2electrode2 = hf2.attrs['cell_index2electrode']
    index2electrode2 = hf2.attrs['index2electrode']

    S2 = np.array(hf2['S'])
    X2 = np.array(hf2['X'])
    Y2 = np.array(hf2['Y'])
    hf2.close()
    
    cell_index2electrode = list()
    cell_index2electrode.extend(cell_index2electrode1)
    cell_index2electrode.extend(cell_index2electrode2)

    index2electrode = list()
    index2electrode.extend(index2electrode1)
    index2electrode.extend(index2electrode2)

    assert np.abs(S1 - S2).sum() == 0
    assert np.abs(Y1 - Y2).sum() == 0

    X = np.hstack([X1, X2])

    hf = h5py.File(output_file, 'w')
    hf['X'] = X
    hf['Y'] = Y1
    hf['S'] = S1
    for aname,aval in list(attrs.items()):
        hf.attrs[aname] = aval
    hf.attrs['cell_index2electrode'] = cell_index2electrode
    hf.attrs['index2electrode'] = index2electrode
    hf.close()


if __name__ == '__main__':

    exp_name = 'GreBlu9508M'
    data_dir = '/auto/tdrive/mschachter/data'
    bird_dir = os.path.join(data_dir, exp_name)
    exp_file = os.path.join(bird_dir, '%s.h5' % exp_name)
    stim_file = os.path.join(bird_dir, 'stims.h5')
    output_dir = os.path.join(bird_dir, 'transforms')
    preproc_dir = os.path.join(bird_dir, 'preprocess')
    agg_dir = os.path.join(data_dir, 'aggregate')

    start_time = None
    end_time = None
    hemis = ['L']

    block_name = 'Site4'
    segment_name = 'Call1'

    seg_uname = '%s_%s_%s' % (block_name, segment_name, ','.join(hemis))
    file_ext = '%s_%s' % (exp_name, seg_uname)
    rep_type = 'raw'
    output_file = os.path.join(output_dir, 'PairwiseCF_%s_%s.h5' % (file_ext, rep_type))

    """
    ######### Code to create the pairwise cf file
    sefile = os.path.join(output_dir, 'StimEvent_%s.h5' % file_ext)
    se = StimEventTransform.load(sefile, rep_types_to_load=[rep_type])
    se.segment_stims(plot=False)

    bs_file = os.path.join(bird_dir, 'transforms', 'BiosoundTransform_%s.h5' % exp_name)
    se.segment_stims_from_biosound(bs_file)

    cft = PairwiseCFTransform()
    cft.transform(se, rep_type=rep_type, debug=False)
    cft.save(output_file)
    """

    cft = PairwiseCFTransform.load(output_file)
    agg_bs_file = os.path.join(agg_dir, 'biosound.h5')

    """
    all_decomps = [('trial_avg_psds',), ('mean_sub_psds',), ('full_psds',),
                   ('trial_avg_psds', 'trial_avg_cfs',),
                   ('mean_sub_psds', 'mean_sub_cfs',),
                   ('full_psds', 'full_cfs',),
                   ('spike_rate',), ('spike_sync',),
                   ('spike_rate', 'spike_sync'),
                  ]
    """

    """
    all_decomps = [('full_psds',), ('spike_rate',)]
    for decomps in all_decomps:
        dstr = '+'.join(decomps)
        ofile = os.path.join(preproc_dir, 'preproc_%s_%s.h5' % (seg_uname, dstr))
        print 'Exporting to %s...' % ofile
        cft.export_for_acoustic_decoder(ofile, decomps, bs_file=agg_bs_file, merge_freqs=True)
    """

    lfp_decomp = 'full'
    lfp_file = os.path.join(preproc_dir, 'preproc_spike+lfp_%s_%s.h5' % (seg_uname, lfp_decomp))
    cft.export_for_lfp_decoder(lfp_file, decomp=lfp_decomp)

    # pfile1 = os.path.join(preproc_dir, 'preproc_Site4_Call1_L_spike_rate.h5')
    # pfile2 = os.path.join(preproc_dir, 'preproc_Site4_Call1_R_spike_rate.h5')
    # ofile = os.path.join(preproc_dir, 'preproc_Site4_Call1_both_spike_rate.h5')
    # merge_preprocs(pfile1, pfile2, ofile)
