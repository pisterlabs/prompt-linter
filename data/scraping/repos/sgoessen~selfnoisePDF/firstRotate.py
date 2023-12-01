#!/usr/bin/env pythonf
'''
!!!!!!!!!! EXPERIMENTAL !!!!!!!!!!!!!!!!

Rotate the vertical component (on both axes) to find the minimum noise.
The theory is when the sensor is at the most upright the vertical will see 
the minimum noise (In the longer periods)

S. Goessen, Guralp Systems, April 2019

'''
import numpy as np
from rotateFunctions import rotateNaxis, rotateEaxis, rotateZaxis
from scipy.signal import coherence as coh
from obspy.signal import PPSD

metadatatest= {
        'poles': [-0.03700796 + 0.03700796j, -0.03700796 - 0.03700796j,
                  -1130.973 + 0j, -1005.3096 + 0j, -502.65482 + 0j],
        'zeros': [0j, 0j],
        'gain': 571507692,
        'sensitivity': (64 * 7700 / 2.872e-6)}

def rotatefirst(stZ, stN, stE):

    increments = [1.0, 0.5, 0.1, 0.05, 0.01]
    start = -5
    end = 5

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
                testz = rotResultZ[0]
                ppsdZ = PPSD(testz.stats, metadatatest)
                ppsdZ.add(rotResultZ)
                theMin, s1, s2, s3 = ppsdZ.extract_psd_values(1000)
                print(i, j, np.average(theMin))
                b.append([np.average(theMin),i,j])

        # Find angle that produces minimum self-noise
        b = np.array(b).T
        idx = np.argmin(b[0])
        if n == len(increments) - 1:
            zResult.append([b[1][idx], b[2][idx]])
            print(b[0][idx],b[1][idx], b[2][idx])
        # Set-up limits for finer grid search
        start_i = b[1][idx] - (increment * 1.1)
        end_i = b[1][idx] + (increment * 1.1)
        start_j = b[2][idx] - (increment * 1.1)
        end_j = b[2][idx] + (increment * 1.1)
    print(zResult[0][0], zResult[0][1])
    return zResult[0][0], zResult[0][1]
