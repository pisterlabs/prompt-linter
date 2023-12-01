#!/usr/bin/env python
'''
Various routines related to estimation of sensor self-noise.
Loosly based on Obspy's PPSD function
Used directly by calc_selfnoise_pdfs.py

Requires Obspy

Stephen Hicks, University of Southampton, 2017
S. Goessen, Guralp Systems, April 2019
'''


import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib.ticker import FormatStrFormatter
from obspy.core.util import AttribDict
from obspy.imaging.cm import obspy_sequential
from obspy.signal.util import prev_pow_2
from obspy.signal.invsim import paz_to_freq_resp
from bisect import bisect
from scipy.signal import coherence as coh


class SelfNoise(object):
    '''
    Class to compile probabilistic power spectral densities for one combination
    of network/station/location/channel/sampling_rate.
    '''
    def __init__(self, sampling_rates, path,
                 db_bins=(-200, -50, 1.), ppsd_length=(0, 0, 0),
                 overlap_hann=7/8, overlap_segment=0.5,
                 period_smoothing_width_octaves=1.0,
                 period_step_octaves=1/8, period_limits=None, **kwargs):
        '''
        Initialize the PPSD object setting all fixed information on the station
        that should not change afterwards to guarantee consistent spectral
        estimates.
        '''

        # save things related to kwargs
        self.db_bins = db_bins
        self.ppsd_length = ppsd_length
        self.path = path
        self.overlap_hann = 7/8
        self.overlap_segment = overlap_segment

        # Calculate: nfft, nlap, array of periods for each sample rate
        # Based on fft setup in McNamara&Buland (13 segments overlapping 75%,
        # truncate to next lower power of 2)
        # print('No. sampling rates =', len(sampling_rates))
        self._sampling_rate = {n: [] for n in range(len(sampling_rates))}
        self._nfft = {n: [] for n in range(len(sampling_rates))}
        self._nlap = {n: [] for n in range(len(sampling_rates))}
        self._len = {n: [] for n in range(len(sampling_rates))}
        self._psd_periods = {n: [] for n in range(len(sampling_rates))}
        self._smoothed_psds_all = {n: [] for n in range(len(sampling_rates))}
        self._spec = {n: [] for n in range(len(sampling_rates))}
        self.hist_stack = {n: [] for n in range(len(sampling_rates))}
        self.smoothedPSDsIndiv = {n: [] for n in range(len(sampling_rates))}
        for n in range(0, len(sampling_rates)):
            self._smoothed_psds_all[n] = {m: [] for m in range(0, 3)}
        for n, sampling_rate in enumerate(sampling_rates):
            # print('sampling_rate = ', sampling_rate)
            self._sampling_rate[n] = sampling_rate
            self._nfft[n] = prev_pow_2(self.ppsd_length[n] *
                                       self._sampling_rate[n])
            #print(self._sampling_rate[n])
            #print(self.ppsd_length[n])
            self._nlap[n] = int(overlap_hann * self._nfft[n])
            self._len[n] = int(self.ppsd_length[n] * self._sampling_rate[n])
            #print('NNFT = ', self._nfft[n])
            #print('% NFFT = ', self._nfft[n]*100/self._len[n])
            #print('% overlap = ', self._nlap[n]*100/self._len[n])
            #print('Nwindows = ',
            #      (self.ppsd_length[n]*sampling_rate /
            #       (self._nfft[n]*(1-self.overlap_hann)))-1)

            # make an initial dummy psd and to get the array of periods
            _, freq = mlab.psd(np.ones(self._len[n]), self._nfft[n],
                               self._sampling_rate[n],
                               noverlap=self.overlap_hann, sides='onesided',
                               scale_by_freq=True)

            freq = freq[1:]  # leave out first entry (offset)
            self._psd_periods[n] = 1.0 / freq[::-1]

        # Merge period arrays from different sample rates
        if len(self._sampling_rate) == 3:
            self._psd_periods_merged = np.concatenate((self._psd_periods[2],
                                                       self._psd_periods[1],
                                                       self._psd_periods[0]))
            period_limits = (self._psd_periods[2][0],
                             self._psd_periods[0][-1])

        elif len(self._sampling_rate) == 2:
            self._psd_periods_merged = np.concatenate((self._psd_periods[1],
                                                       self._psd_periods[0]))
            period_limits = (self._psd_periods[1][0],
                             self._psd_periods[0][-1])

        period_step_factor = 2 ** period_step_octaves
        period_smoothing_width_factor = 2 ** period_smoothing_width_octaves

        # calculate left/right edge and center of first period bin
        per_left = period_limits[0] / (period_smoothing_width_factor ** .5)
        per_right = per_left * period_smoothing_width_factor
        per_center = math.sqrt(per_left * per_right)

        # build lists
        per_octaves_left = [per_left]
        per_octaves_right = [per_right]
        per_octaves_center = [per_center]

        # do this for the whole period range and append the values to our lists
        while per_center < period_limits[1]:
            per_left *= period_step_factor
            per_right = per_left * period_smoothing_width_factor
            per_center = math.sqrt(per_left * per_right)
            per_octaves_left.append(per_left)
            per_octaves_right.append(per_right)
            per_octaves_center.append(per_center)
        per_octaves_left = np.array(per_octaves_left)
        per_octaves_right = np.array(per_octaves_right)
        per_octaves_center = np.array(per_octaves_center)
        valid = per_octaves_right > self._psd_periods_merged[0]
        valid &= per_octaves_left < self._psd_periods_merged[-1]
        per_octaves_left = per_octaves_left[valid]
        per_octaves_right = per_octaves_right[valid]
        per_octaves_center = per_octaves_center[valid]
        self.period_bins = np.vstack([
            # left edge of smoothing (for calculating the bin value from psd
            per_octaves_left,
            # left xedge of bin (for plotting)
            per_octaves_center / (period_step_factor ** 0.5),
            # bin center (for plotting)
            per_octaves_center,
            # right xedge of bin (for plotting)
            per_octaves_center * (period_step_factor ** 0.5),
            # right edge of smoothing (for calculating the bin value from psd
            per_octaves_right])

        # Set up the binning for the db scale.
        num_bins = int((db_bins[1] - db_bins[0]) / db_bins[2])
        self.dBedges = np.linspace(db_bins[0], db_bins[1], num_bins + 1,
                                   endpoint=True)
        self._db_bin_centers = ((self.dBedges[:-1] +
                                 self.dBedges[1:]) / 2.0)

        # Find point to merge intersections
        self._idx_intersect_1 = bisect(self.period_bins[2, :], 100)

        if len(self._sampling_rate) == 3:
            self._idx_intersect_2 = bisect(self.period_bins[2, :], 16.384)

    def SNadd(self, streamall):
        '''
        Process all traces with compatible information and add their spectral
        estimates to the histogram containing the probabilistic psd.
        Add extra sample onto day to cut out full day.
        '''
        for n in range(0, len(streamall)):
#            print('Processing stream: '
#                  '{0} sps'.format(streamall[n][0][0].stats.sampling_rate))
            stream = streamall[n][0][0]
            t1 = stream.stats.starttime
            t2 = stream.stats.endtime + (1 / self._sampling_rate[n])
            while t1 + self.ppsd_length[n] <= t2:
                n_stream = 1
                trace_all = []
                for n_stream, stream in enumerate(streamall[n]):
                    slice = stream[0].slice(t1, t1 + self.ppsd_length[n])
                    #if n_stream == 2:
                    #    print('Processing: {0} - {1}'
                    #          .format(slice.stats.starttime.ctime(),
                    #                  slice.stats.endtime.ctime()))
                    if len(stream[0]) == self._len[n] + 1:
                        slice.data = slice.data[:-1]
                    trace_all.append(slice)

                    # Calculate self-noise when 3 traces have been loaded in
                    if n_stream == 2:
                        self.__process_SN(trace_all, n)
                # Advance to next segment
                t1 += (1 - self.overlap_segment) * self.ppsd_length[n]

    def __process_SN(self, trace_all, n):
        '''
        Process the self-noise
        '''
        # First, calculate the auto-power spectra, then the cross-power spectra
        # N.B. A linear detrend for some reason exaggerates the noise in the
        # microseismic peak (ignores any alignment done during processing)
        p11, _freq = mlab.csd(trace_all[0].data, trace_all[0].data,
                              self._nfft[n],
                              self._sampling_rate[n],
                              detrend='constant',
                              window=mlab.window_hanning,
                              noverlap=self._nlap[n],
                              sides='onesided', scale_by_freq=True)
        p22, _freq = mlab.csd(trace_all[1].data, trace_all[1].data,
                              self._nfft[n],
                              self._sampling_rate[n],
                              detrend='constant',
                              window=mlab.window_hanning,
                              noverlap=self._nlap[n],
                              sides='onesided', scale_by_freq=True)
        p33, _freq = mlab.csd(trace_all[2].data, trace_all[2].data,
                              self._nfft[n],
                              self._sampling_rate[n],
                              detrend='constant',
                              window=mlab.window_hanning,
                              noverlap=self._nlap[n],
                              sides='onesided', scale_by_freq=True)

        # Now the cross-power spectra

        #p13 = mlab.coh(p11,p33,)
        p13, _freq = mlab.csd(trace_all[0].data, trace_all[2].data,
                              self._nfft[n], self._sampling_rate[n],
                              detrend='constant',
                              window=mlab.window_hanning,
                              noverlap=self._nlap[n], sides='onesided',
                              scale_by_freq=True)
        p21, _freq = mlab.csd(trace_all[1].data, trace_all[0].data,
                              self._nfft[n], self._sampling_rate[n],
                              detrend='constant',
                              window=mlab.window_hanning,
                              noverlap=self._nlap[n], sides='onesided',
                              scale_by_freq=True)
        p23, _freq = mlab.csd(trace_all[1].data, trace_all[2].data,
                              self._nfft[n], self._sampling_rate[n],
                              detrend='constant',
                              window=mlab.window_hanning,
                              noverlap=self._nlap[n], sides='onesided',
                              scale_by_freq=True)
        
        n11 = p11 - p21 * p13 / p23
        n22 = p22 - np.conjugate(p23) * p21 / np.conjugate(p13)
        n33 = p33 - p23 * np.conjugate(p13) / p21
        sn = []
        sn.append(n11)
        sn.append(n22)
        sn.append(n33)
        

        # Loop over self noise arrays
        # Leave out the first entry (offset) & reverse for period
                                        # Note lines removed.. these were removing the response
                                        # from the data using a pole zero file.
                                        # Now the raw data has already had the response/FIR filters removed
                                        # by using the remove response function of Obspy.
        for m in range(0, 3):
            sn[m] = sn[m][1:]
            sn[m] = sn[m][::-1]
                                        #resp = self._get_response_from_paz_dict(slice, n)
                                        #resp = resp[1:]
                                        #resp = resp[::-1]
                                        #respamp = np.absolute(resp * np.conjugate(resp))

            w = 2.0 * math.pi * _freq[1:]
            w = w[::-1]
                                        #if self.sensor_type == 'accelerometer':
                                        #    spec = sn[m] / respamp
                                        #elif self.sensor_type == 'seismometer':
                                        #    spec = (w ** 2) * sn[m] / respamp
            spec = (w ** 2) * sn[m]


            # go to dB
            self._spec = 10 * np.log10(np.abs(spec))
            smoothed_psd = []
            #print(self.period_bins)
            for per_left, per_right in zip(self.period_bins[0, :],
                                           self.period_bins[4, :]):
                specs = self._spec[(per_left <= self._psd_periods[n]) &
                                   (self._psd_periods[n] <= per_right)]
                smoothed_psd.append(np.nanmean(specs))
            
            smoothed_psd = np.array(smoothed_psd, dtype=np.float32)

            # Change any NaNs or Infs to -150: dummy for outside range of PSD
            idx_nan = np.isnan(smoothed_psd)
            smoothed_psd[idx_nan] = -150
            idx_nan2 = np.isinf(smoothed_psd)
            smoothed_psd[idx_nan2] = -150
            self._smoothed_psds_all[n][m].append(smoothed_psd)

    def SNplot(self, filename=None, show_coverage=True, show_histogram=True,
               show_percentiles=False, percentiles=[0, 25, 50, 75, 100],
               show_noise_models=True, grid=True, show=True,
               max_percentage=None, period_lim=(0.02, 1000), show_mode=False,
               show_mean=False, cmap=obspy_sequential,
               xaxis_frequency=False, sampling_rates=(1,40,100), sensors=[],
               sensor_model='', start_date='', end_date=''):
        for m in range(0, len(sensors)):
            fig = plt.figure(figsize=(12, 10))
            fig.ppsd = AttribDict()
            ax = fig.add_subplot(111)

            # Make empty database
            for n in range(0, len(sampling_rates)):

                #print('No. PSD segments for sensor {0}, sample rate {1} = {2}'
                #      .format(m, n, len(self._smoothed_psds_all[n][m])))

                self.hist_stack[n] = np.zeros((len(self.period_bins[2, :]),
                                               len(self._db_bin_centers)),
                                              dtype=np.float32)

                self.hist_stack_com = np.zeros((len(self.period_bins[2, :]),
                                                len(self._db_bin_centers)),
                                               dtype=np.float32)

                self.smoothedPSDsIndiv[n] = (
                    np.array(self._smoothed_psds_all[n][m]))
                for i, period_bin in enumerate(self.smoothedPSDsIndiv[n].T):
                    self.hist_stack[n][i, :], _ = np.histogram(period_bin,
                                                               bins=self.
                                                               dBedges)

            # Concatenatate stacks based on idx_intersect
            if len(self._sampling_rate) == 3:
                for i, period_bin in enumerate(self.hist_stack[2]):
                    if i < self._idx_intersect_2:
                        self.hist_stack_com[i, :] = period_bin
                for i, period_bin in enumerate(self.hist_stack[1]):
                    if (i >= self._idx_intersect_2 and
                       i < self._idx_intersect_1):
                        self.hist_stack_com[i, :] = period_bin
                for i, period_bin in enumerate(self.hist_stack[0]):
                    if i >= self._idx_intersect_1:
                        self.hist_stack_com[i, :] = period_bin

            if len(self._sampling_rate) == 2:
                for i, period_bin in enumerate(self.hist_stack[1]):
                    if i < self._idx_intersect_1:
                        self.hist_stack_com[i, :] = period_bin
                for i, period_bin in enumerate(self.hist_stack[0]):
                    if i >= self._idx_intersect_1:
                        self.hist_stack_com[i, :] = period_bin

            # At each frequency, calculate preliminary mode, find indexes of
            # dB bins that are outliers (30 dB above mode), set stacks to zero
            mode_all = self._db_bin_centers[self.hist_stack_com.argmax(axis=1)]
            for i, mode_value in enumerate(mode_all):
                jdxs = [j for j, k in enumerate(self._db_bin_centers)
                        if k > 30 + mode_value]
                for jdx in jdxs:
                    if self.hist_stack_com[i][jdx] > 0.0:
                        self.hist_stack_com[i][jdx] = 0.0

            # Normalise histogram to sum of each bin hits
            self.hist_stack_norm = np.zeros((len(self.period_bins[2, :]),
                                             len(self._db_bin_centers)),
                                            dtype=np.float32)
            norm = self.hist_stack_com.sum(axis=1)
            self.hist_stack_norm = (100*self.hist_stack_com.T/norm).T

            # Calculate and plot percentiles
            xdata = 1.0 / self.period_bins[2, :]
            for percentile in percentiles:
                percentile_value = np.zeros((len(self.period_bins[2, :]), 1))
                percentile_value = np.percentile(self.smoothedPSDsIndiv[n].T,
                                                 percentile, axis=1)
                ax.plot(xdata, percentile_value, color='green', zorder=8,
                        label='{0}st percentile'.format(percentile))

            # Calculate minimum self-noise
            min_selfnoise = np.zeros((len(self.period_bins[2, :]), 1))
            for i, period_bin in enumerate(self.hist_stack_norm):
                jdx = next((j for j, x in enumerate(period_bin) if x != 0),
                           None)
                min_selfnoise[i] = self._db_bin_centers[jdx]
            xdata = 1.0 / self.period_bins[2, :]
            ax.plot(xdata, min_selfnoise, color='lawngreen', zorder=9,
                    label='Minimum self-noise', linewidth=2)

            # Write minimum self-noise to file
            if show_mode is True:
                file = open(r'{0}\{1}_{2}_{3}_{4}_minselfnoise.txt'
                            .format(self.path, sensor_model, sensors[m], start_date,
                                    end_date),'w')
                for x, y in zip(xdata, min_selfnoise):
                    file.write('{0:7.4f} {1:7.4f}\n'
                               .format(float(x), float(y)))
                file.close()

            # Re-calculate and plot mode
            mode = self._db_bin_centers[self.hist_stack_norm.argmax(axis=1)]
            xdata = 1.0 / self.period_bins[2, :]
            ax.plot(xdata, mode, color='blue', zorder=9, label=('Modal '
                                                                'self-noise'))

            # Write mode self noise to file
            if show_mode is True:
                file = open(r'{0}\{1}_{2}_{3}_{4}_modeselfnoise.txt'
                            .format(self.path, sensor_model, sensors[m], start_date,
                                    end_date),'w')
                for x, y in zip(xdata, mode):
                    file.write('{0:8.5f} {1:8.5f}\n'
                               .format(float(x), float(y)))
                file.close()

            data = np.load('noise_models.npz')
            periods = data['model_periods']
            xdata = 1.0 / periods
            nlnm = data['low_noise']
            ax.plot(xdata, nlnm, '0.4', linewidth=2, zorder=10,
                    label='NLNM')
            nhnm = data['high_noise']
            ax.plot(xdata, nhnm, color='peru', linewidth=2, zorder=10,
                    label='NHNM')

            fig.ppsd.cmap = cmap
            fig.ppsd.label = '[%]'
            fig.ppsd.max_percentage = 100
            fig.ppsd.grid = grid
            fig.ppsd.xaxis_frequency = xaxis_frequency
            fig.ppsd.color_limits = (0, fig.ppsd.max_percentage)

            ax = fig.axes[0]
            xlim = ax.get_xlim()
            data = self.hist_stack_norm

            xedges = np.concatenate([self.period_bins[1, 0:1],
                                     self.period_bins[3, :]])
            xedges = 1/xedges

            fig.ppsd.meshgrid = np.meshgrid(xedges, self.dBedges)
            X, Y = fig.ppsd.meshgrid
            ppsd = ax.pcolormesh(X, Y, data.T, cmap=fig.ppsd.cmap, zorder=-1)
            fig.ppsd.quadmesh = ppsd
            cb = plt.colorbar(ppsd, ax=ax)
            cb.set_label('Probability (%)')
            cb.set_clim(*fig.ppsd.color_limits)
            ppsd.set_clim(*fig.ppsd.color_limits)
            fig.ppsd.colorbar = cb

            color = {}
            ax.grid(b=True, which='major', **color)
            ax.grid(b=True, which='minor', **color)

            ax.set_xlim(*xlim)
            ax.semilogx()
            xlim = map(lambda x: 1.0 / x, period_lim)
            ax.set_xlabel('Frequency [Hz]')
            ax.invert_xaxis()
            ax.set_xlim(sorted(xlim))
            ax.set_ylim(self.dBedges[0], self.dBedges[-1])
            ax.set_ylabel('Amplitude [$m^2/s^4/Hz$] [dB]')
            ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
            legend = ax.legend()
            legend.get_frame().set_facecolor('0.95')
            legend.set_zorder(20)
            ax.set_title('Sensor: {0} (serial number: T3{1})\n'
                         'Date range: {2}/{3}/{4} - {5}/{6}/{7}\n'
                         'Number of PSD segments: LH={8}; HH={9}'
                         .format(sensor_model, sensors[m], start_date[0:4],
                                 start_date[4:6], start_date[6:8],
                                 end_date[0:4], end_date[4:6], end_date[6:8],
                                 len(self._smoothed_psds_all[0][m]),
                                 len(self._smoothed_psds_all[1][m])))
            if show_mode is True:
                plt.savefig(r'{0}\{1}_{2}_{3}_{4}.png'
                            .format(self.path, sensor_model, sensors[m], start_date,
                                    end_date), format='png', dpi=800, size=1000)
                print('Outputing result to {0}\{1}_{2}_{3}_{4}.png'
                            .format(self.path, sensor_model, sensors[m], start_date,
                                    end_date))
                #plt.savefig(r'c:\Python\Selfnoise\Results\{0}_{1}_{2}_{3}.eps'
                #            .format(sensor_model, sensors[m], start_date,
                #                    end_date), format='eps', dpi=800)

            elif show_mode is False:
                plt.savefig(os.path.join(r'{0}\tmp_{1}_{2}.png'.format(self.path, m,
                            sensor_model)))
                plt.close(fig)
