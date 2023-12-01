#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2019, 2020, 2021 Johannes Demel.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np

from .coherence import ChannelCoherenceRappaport
from .powerdelayprofile import FFTPowerDelayProfile, PowerDelayProfile
from .awgn import AWGN
from .timevariantchannel import CoherentTimeVariantChannel, TimeVariantChannel, TransmissionChannel
from .frequencydomainchannel import FrequencyDomainChannel


class RayleighAWGNSimulationChannel(object):
    """RayleigAWGNSimulationChannel
    awgn_channel: One channel that's stateless
    fading_channels: a matrix of rx_ antennas x tx_antennas identical rayleigh channels
    """

    def __init__(self, awgn_channel, fading_channels):
        self._awgn_channel = awgn_channel
        self._fading_channels = fading_channels

    def state(self):
        s = self._awgn_channel.state()
        s.update(self._fading_channels[0][0].state())
        s['tx_antennas'] = self.tx_antennas()
        s['rx_antennas'] = self.rx_antennas()
        return s

    def step(self, time_delta=1.e-3):
        for channel_row in self._fading_channels:
            for chan in channel_row:
                chan.step(time_delta)

    def snr(self):
        return self._awgn_channel.snr()

    def channel_taps(self):
        return [[t.channel_taps() for t in row] for row in self._fading_channels]

    def tx_antennas(self):
        return len(self._fading_channels[0])

    def rx_antennas(self):
        return len(self._fading_channels)

    def channel_dimensions(self):
        return (self.rx_antennas(), self.tx_antennas())

    def channel_length(self):
        return self._fading_channels[0][0].time_domain_length()

    def transmit(self, tx_symbols):
        res = []
        for chans in self._fading_channels:
            r = np.sum((c.transmit(s)
                        for c, s in zip(chans, tx_symbols)), axis=0)
            res.append(self._awgn_channel.transmit(r))
        return res


class FrequencyDomainEqualizer(object):
    def __init__(self, equalizer_type='MF'):
        equalizer_type = equalizer_type.upper()
        assert equalizer_type in ('MF', 'ZF', 'MMSE')
        self._equalizer_type = equalizer_type
        self._equalizer_func = self.equalize_matched_filter
        if equalizer_type == 'ZF':
            self._equalizer_func = self.equalize_zero_forcing
        elif equalizer_type == 'MMSE':
            self._equalizer_func = self.equalize_minimum_mean_squared_error

    def state(self):
        return {'equalizer_type': self._equalizer_type}

    def equalize_matched_filter(self, rx_mod, fd_channel_taps, _):
        return rx_mod * np.conj(fd_channel_taps)

    def equalize_zero_forcing(self, rx_mod, fd_channel_taps, _):
        return rx_mod / fd_channel_taps

    def equalize_minimum_mean_squared_error(self, rx_mod, fd_channel_taps, variance):
        ctaps = np.conj(fd_channel_taps)
        return rx_mod * ctaps / (ctaps * fd_channel_taps + variance)

    def equalize(self, rx_mod, fd_channel_taps, variance):
        return self._equalizer_func(rx_mod, fd_channel_taps, variance)
        # if self._equalizer_type == 'MF':
        #     return rx_mod * np.conj(fd_channel_taps)
        # elif self._equalizer_type == 'ZF':
        #     return rx_mod / fd_channel_taps
        # elif self._equalizer_type == 'MMSE':
        #     ctaps = np.conj(fd_channel_taps)
        #     return rx_mod * ctaps / (ctaps * fd_channel_taps + variance)


class FrequencyDomainRayleighChannel:
    """Channel to simulate transmission over a Rayleigh channel in Frequency Domain"""

    def __init__(self, awgn_channel, fading_channel, equalizer_type='MF'):
        self._awgn_channel = awgn_channel
        self._channel = fading_channel
        self._equalizer = FrequencyDomainEqualizer(equalizer_type)
        self._frequency_domain_gains = self._channel.freq_domain_gains()
        self._frequency_domain_taps = self._channel.freq_domain_taps()

    def state(self):
        s = self._awgn_channel.state()
        s.update(self._channel.state())
        s.update(self._equalizer.state())
        return s

    def channel_taps(self):
        return self._channel.time_domain_taps()

    def channel_length(self):
        return self._channel.time_domain_length()

    def frequeny_domain_taps(self):
        return self._frequency_domain_taps

    def channel_gains(self):
        return self._frequency_domain_gains

    def step(self, time_delta=1.e-3):
        self._channel.step(time_delta)

    def snr(self):
        return self._awgn_channel.snr()

    def transmit(self, tx_mod):
        reps = int(np.ceil(1. * tx_mod.size / self._channel.subcarriers()))
        self._frequency_domain_taps = np.tile(
            self._channel.freq_domain_taps(), reps)
        self._frequency_domain_gains = np.tile(
            self._channel.freq_domain_gains(), reps)
        rx_mod = tx_mod * self._frequency_domain_taps
        rx_mod = self._awgn_channel.transmit(rx_mod)
        rx_mod = self._equalizer.equalize(
            rx_mod, self._frequency_domain_taps, self._awgn_channel.variance())
        return rx_mod


class ChannelFactory:
    def __init__(self, channel_domain, channel_type, effective_rate,
                 subcarriers=1, rms_delay_spread=46.e-9, max_delay_spread=250.e-9,
                 bandwidth=20.e6, carrier_frequency=None, velocity=None,
                 tx_antennas=1, rx_antennas=1, snr_mode='ebn0', equalizer_type='MF'):
        self._snr_db = 0.0  # A dummy to carry init member in ctor
        snr_mode = snr_mode.lower()
        assert snr_mode in ('ebn0', 'edn0')
        self._snr_mode = snr_mode
        self._channel_domain = channel_domain.lower()
        assert self._channel_domain in ('time', 'frequency')
        self._channel_type = channel_type.lower()
        assert self._channel_type in ('awgn', 'rayleigh')
        self._effective_rate = effective_rate
        self._subcarriers = subcarriers
        self._rms_delay_spread = rms_delay_spread
        self._max_delay_spread = max_delay_spread
        self._bandwidth = bandwidth
        self._carrier_frequency = carrier_frequency
        self._velocity = velocity
        self._tx_antennas = tx_antennas
        self._rx_antennas = rx_antennas
        if channel_domain == 'time' and equalizer_type != 'ZF':
            raise f'Channel domain: {channel_domain} does not support "{equalizer_type}" equalizer!'
        self._equalizer_type = equalizer_type

    def state(self):
        tmp = vars(self)
        s = {k[1:]: v for k, v in tmp.items()}
        return s

    def __str__(self):
        s = self.state()
        s = type(self).__name__
        s += '('
        fields = [f'{k}={v}' for k, v in self.state().items()]
        s += ', '.join(fields)
        return s + ')'

    def set_snr(self, snr_db):
        self._snr_db = snr_db

    def snr(self):
        return self._snr_db

    def _create_awgn(self):
        if self._channel_domain in 'time':
            awgnsc = self._subcarriers
        else:
            awgnsc = 1
        eff_rate = self._effective_rate
        if self._snr_mode == 'edn0':
            eff_rate = 1.
        return AWGN(self._snr_db, eff_rate, awgnsc)

    def _create_rayleigh(self):
        if self._channel_domain in 'frequency':
            pdp = FFTPowerDelayProfile(self._rms_delay_spread,
                                       self._max_delay_spread, self._bandwidth,
                                       self._subcarriers)
        else:
            scale = 1. / np.sqrt(1. * self._tx_antennas * self._rx_antennas)
            pdp = PowerDelayProfile(self._rms_delay_spread,
                                    self._max_delay_spread, self._bandwidth, scale=scale)
        if self._carrier_frequency is not None and self._velocity is not None:
            coherence = ChannelCoherenceRappaport(
                self._carrier_frequency, self._velocity)
            channel = CoherentTimeVariantChannel(pdp, coherence)
        else:
            channel = TimeVariantChannel(pdp)
        return channel

    def create(self, snr_db=None):
        if snr_db is not None:
            self.set_snr(snr_db)
        channel = awgn_channel = self._create_awgn()

        if self._channel_type in 'rayleigh':
            if self._channel_domain in 'time':
                fading_channels = [[TransmissionChannel(self._create_rayleigh()) for _ in range(
                    self._tx_antennas)] for _ in range(self._rx_antennas)]
                channel = RayleighAWGNSimulationChannel(awgn_channel,
                                                        fading_channels)
            else:
                fading_channel = FrequencyDomainChannel(self._create_rayleigh())
                channel = FrequencyDomainRayleighChannel(
                    awgn_channel, fading_channel, self._equalizer_type)
        return channel
