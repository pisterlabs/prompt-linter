#!/usr/bin/env pythonf
'''
Function to measure coherence between two sensors.
Rotates second sensor to achive maxcoherence and returns that result in degrees.
Intended to be used for self-noise probablility density.

Requires Obspy

S. Hicks, University of Southampton, May-July 2017
S. Goessen, Guralp Systems, April 2019
'''

import numpy as np
from rotateFunctions import rotateNaxis, rotateEaxis, rotateZaxis
from scipy.signal import coherence as coh


def rotateAll(strefZ, strefN, strefE, stZ, stN, stE): # stref? is the reference,
                            # st? is the sensor of which the rotation is applied
    '''
    Function takes 3 components from 2 sensors and returns the angle at which to rotate the second sensor for maximum coherence.
    Completes this on all 3 components separately therefore returns 6 angles.
    '''

    #~#~#~#~#~#~# START OF USER ADJUSTABLE VARIABLES #~#~#~#~#~#~#~#~#
    increments = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001] 
                                # len(increments) = Number of iterations
    start = -5 # Minimum angle to search
    end = 5 # Maximium angle to search
    f1 = 0.15 # Lower of freqency range in Hz
    f2 = 0.24 # Lower of freqency range in Hz
    #f1 and f2 can be adjusted to centre around the ocean microseism
    samplerate = 1 # Sample rate of data
    #~#~#~#~#~#~# END OF USER ADJUSTABLE VARIABLES #~#~#~#~#~#~#~#~#

# Rotates the Z output = zResult
    start_i = start
    end_i = end
    start_j = start
    end_j = end
    zResult = []
    rotResultZ = []

    for n, increment in enumerate(increments):
        b = []
        idx = []
        for i in np.arange(start_i, end_i, increment):
            for j in np.arange(start_j, end_j, increment):
                rotResultZ = rotateZaxis(stE, stN, stZ, i, j)
                f, COH = coh(strefZ[0].data, rotResultZ[0].data, fs=samplerate,
                             window='hanning')

                b.append([i, j, np.mean(COH[(f > f1) * (f < f2)])])
                
        # Find angle that produces minimum self-noise
        b = np.array(b).T
        idx = np.argmax(b[2])
        if n == len(increments) - 1:
            zResult.append([b[0][idx], b[1][idx]])

        # Set-up limits for finer grid search
        start_i = b[0][idx] - (increment * 1.1)
        end_i = b[0][idx] + (increment * 1.1)
        start_j = b[1][idx] - (increment * 1.1)
        end_j = b[1][idx] + (increment * 1.1)

    start_i = start             # Rotates the N/S output = nResult
    end_i = end
    start_j = start
    end_j = end
    nResult = []
    rotResultN = []

    for n, increment in enumerate(increments):
        b = []
        for i in np.arange(start_i, end_i, increment):
            for j in np.arange(start_j, end_j, increment):
                rotResultN = rotateNaxis(stE, stN, stZ, i, j)
                f, COH = coh(strefN[0].data, rotResultN[0].data, fs=1,
                             window='hanning')

                b.append([i, j, np.mean(COH[(f > f1) * (f < f2)])])

        b = np.array(b).T
        idx = np.argmax(b[2])
        if n == len(increments) - 1:
            nResult.append([b[0][idx], b[1][idx]])

        # Set-up limits for finer grid search
        start_i = b[0][idx] - (increment * 1.1)
        end_i = b[0][idx] + (increment * 1.1)
        start_j = b[1][idx] - (increment * 1.1)
        end_j = b[1][idx] + (increment * 1.1)

    start_i = start  # Rotates the E/W output = eResult
    end_i = end
    start_j = start
    end_j = end
    eResult = []
    rotResultE = []
    for n, increment in enumerate(increments):
        b = []
        for i in np.arange(start_i, end_i, increment):
            for j in np.arange(start_j, end_j, increment):
                rotResultE = rotateEaxis(stE, stN, stZ, i, j)
                f, COH = coh(strefE[0].data, rotResultE[0].data, fs=samplerate,
                             window='hanning')

                b.append([i, j, np.mean(COH[(f > f1) * (f < f2)])])
                # print(i, j, np.mean(COH[(f > f1)*(f < f2)]))

        # Find angle that produces minimum self-noise
        b = np.array(b).T
        idx = np.argmax(b[2])
        if n == len(increments) - 1:
            eResult.append([b[0][idx], b[1][idx]])
        # Set-up limits for finer grid search
        start_i = b[0][idx] - (increment * 1.1)
        end_i = b[0][idx] + (increment * 1.1)
        start_j = b[1][idx] - (increment * 1.1)
        end_j = b[1][idx] + (increment * 1.1)

    return (zResult[0][0], zResult[0][1],
            nResult[0][0], nResult[0][1],
            eResult[0][0], eResult[0][1])
