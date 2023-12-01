#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2019, 2020 Johannes Demel.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np

from coherence import ChannelCoherenceJakes, ChannelCoherenceRappaport


def plot_channel_coherence_distance():
    import matplotlib.pyplot as plt
    modelJakes = ChannelCoherenceJakes(3.8e9, 15.)
    modelRappaport = ChannelCoherenceRappaport(3.8e9, 15.)
    n = np.arange(0, 1., 1e-3)
    jakes = modelJakes.get_covariance_distance(n)
    rappaport = modelRappaport.get_covariance_distance(n)
    plt.plot(n, jakes, label='Jakes')
    plt.plot(n, rappaport, label='Rappaport')
    plt.xlabel('distance [normalized to wavelength]')
    plt.ylabel('covariance')
    plt.grid()
    plt.legend()
    plt.show()


def plot_channel_coherence_time(f=3.8e9, v=15.):
    import matplotlib.pyplot as plt
    t = np.arange(0., 3e-3, 1e-6)

    modelJakes = ChannelCoherenceJakes(f, v)
    modelRappaport = ChannelCoherenceRappaport(f, v)
    print('Channel covariance after 1ms {}'.format(
        modelJakes.get_covariance_time(1.e-3)))

    jakes = modelJakes.get_covariance_time(t)
    rappaport = modelRappaport.get_covariance_time(t)
    t *= 1e3
    plt.plot(t, jakes, label='Jakes')
    plt.plot(t, rappaport, label='Rappaport')
    plt.grid()
    plt.legend()
    plt.title('Channel coherence time for v={}m/s and fc={}GHz'.format(
        v, 1. * f / 1e9))
    plt.xlabel('time [ms]')
    plt.ylabel('covariance')
    plt.show()


def main():
    fc = 3.8e9
    v = 15.
    plot_channel_coherence_distance()
    plot_channel_coherence_time(fc, v)


if __name__ == '__main__':
    main()
