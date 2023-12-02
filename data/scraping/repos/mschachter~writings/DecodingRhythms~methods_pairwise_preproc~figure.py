import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lasp.signal import coherency

from methods_preproc.figure import get_full_data
from utils import set_font, get_this_dir, get_full_data


def draw_figures():

    d = get_full_data('GreBlu9508M', 'Site4', 'Call1', 'L', 287)

    syllable_index = 1

    syllable_start = d['syllable_props'][syllable_index]['start_time'] - 0.020
    syllable_end = d['syllable_props'][syllable_index]['end_time'] + 0.030

    sr = d['lfp_sample_rate']
    lfp_mean = d['lfp'].mean(axis=0)
    lfp_t = np.arange(lfp_mean.shape[1]) / sr
    nelectrodes,nt = lfp_mean.shape
    lfp_i = (lfp_t >= syllable_start) & (lfp_t <= syllable_end)

    electrode_order = d['electrode_order']

    # compute the cross coherency between each pair of electrodes
    nelectrodes = 16
    lags = np.arange(-20, 21)
    lags_ms = (lags / sr)*1e3
    nlags = len(lags)
    window_fraction = 0.60
    noise_floor_db = 25,
    cross_coherency = np.zeros([nelectrodes, nelectrodes, nlags])

    for i in range(nelectrodes):
        for j in range(nelectrodes):
            if i == j:
                continue

            lfp1 = lfp_mean[i, lfp_i]
            lfp2 = lfp_mean[j, lfp_i]

            cross_coherency[i, j, :] = coherency(lfp1, lfp2, lags, window_fraction=window_fraction, noise_floor_db=noise_floor_db)

    figsize = (24, 13)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(top=0.95, bottom=0.02, right=0.97, left=0.03, hspace=0.20, wspace=0.20)

    gs = plt.GridSpec(nelectrodes, nelectrodes)

    for k in range(nelectrodes):
        for j in range(k):
            ax = plt.subplot(gs[k, j])
            plt.axhline(0, c='k')
            plt.plot(lags_ms, cross_coherency[k, j], 'k-', linewidth=2.0, alpha=0.8)
            plt.axis('tight')
            plt.ylim(-.25, 0.5)

            plt.yticks([])
            plt.xticks([])

            if k == nelectrodes-1:
                plt.xlabel('E%d' % electrode_order[j])
                xtks = [-40, 0, 40]
                plt.xticks(xtks, ['%d' % x for x in xtks])
            if j == 0:
                plt.ylabel('E%d' % electrode_order[k])
                ytks = [-0.2, 0.4]
                plt.yticks(ytks, ['%0.1f' % x for x in ytks])

    ax = plt.subplot(gs[:7, (nelectrodes-8):])

    voffset = 5
    for n in range(nelectrodes):
        plt.plot(lfp_t, lfp_mean[nelectrodes-n-1, :] + voffset*n, 'k-', linewidth=3.0, alpha=0.75)
    plt.axis('tight')
    ytick_locs = np.arange(nelectrodes) * voffset
    plt.yticks(ytick_locs, list(reversed(d['electrode_order'])))
    plt.ylabel('Electrode')
    plt.axvline(syllable_start, c='k', linestyle='--', linewidth=3.0, alpha=0.7)
    plt.axvline(syllable_end, c='k', linestyle='--', linewidth=3.0, alpha=0.7)
    plt.xlabel('Time (s)')

    fname = os.path.join(get_this_dir(), 'figure.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


if __name__ == '__main__':
    set_font()
    draw_figures()








