#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2015
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
import scipy.signal
from FDMT import CoherentDedispersion, DispersionConstant, STFT

__author__ = "Keith Bannister <keith.bannister@csiro.au>"


def dispersed_voltage(f_min, f_max, n_pulse, n_samps, D=5, PulseSig = 0.4, PulsePosition = 4134567/4):
    '''
    Produces a dispersed voltage time zeries
    Modified from FDMT.py
    '''
    print(locals())
    N_total = n_samps
    PulseLength = n_pulse
    practicalD = DispersionConstant * D
    I = np.random.normal(0,1,N_total)
    I[PulsePosition:PulsePosition+PulseLength] += np.random.normal(0,PulseSig,PulseLength)
    print("MAX Thoretical SNR:", np.sum(np.abs(I[PulsePosition:PulsePosition+PulseLength])**2 - np.mean(abs(I)**2)) / (np.sqrt(PulseLength*np.var(abs(I)**2))))
    
    X = CoherentDedispersion(I, -D, f_min,f_max,False)    

    return X

def stft(X, N_f, N_t, N_bins):
    ''' Copied from FDMT.py'''
    XX = np.abs(np.fft.fft(X.reshape([N_f,N_t*N_bins]),axis = 1))**2
    XX = np.transpose(XX)
    XX = np.sum(XX.reshape(N_f,N_bins,N_t),axis=1)
    
    E = np.mean(XX[:,:10])
    XX -= E
    V = np.var(XX[:,:10])
    
    XX /= (0.25*np.sqrt(V))
    V = np.var(XX[:,:10])

    return XX



def dispersed_stft(f_min, f_max, N_t, N_f, N_bins, D=5, PulseSig=0.4, PulsePosition=4134567/4):
    n_pulse = N_f * N_bins
    n_samps = N_t * n_pulse
    x = dispersed_voltage(f_min, f_max, n_pulse, n_samps, D, PulseSig, PulsePosition)
    return stft(x, N_f, N_t, N_bins)

def resample(x, axis=0, window=('kaiser', 5.0)):
    '''
    Simulates (roughly) the ASKAP filterbank
    TODO: load ASKAP coefficients and do it properly
    :x: Array to resample (should complex)
    :axis: Axis of x to filterbank
    :window: Desired window to use or FIR filter coefficients
    :@see: scipy.signal.resample_poly
    
    '''
    assert np.iscomplex(x)
    out = scipy.signal.resample_poly(32, 27, axis, window)

    return out

def detect(x, N):
    '''
    Squares and block averages by N
    
    '''
    assert N > 0
    
    x = np.real(x*np.conj(x))
    
        



def _main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Script description')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    

if __name__ == '__main__':
    _main()
