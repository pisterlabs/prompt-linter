#!/usr/bin/env python

'''Miscellaneous functions

'''

import numpy as np
import scipy.constants


def signal_delay(st1, st2, ecef):
    '''Signal delay due to speed of light between station-1 to ecef position to station-2
    '''
    r1 = np.linalg.norm(ecef - st1.ecef[:,None], axis=0)
    r2 = np.linalg.norm(ecef - st1.ecef[:,None], axis=0)
    dt = (r1 + r2)/scipy.constants.c
    return dt


def instantaneous_to_coherrent(gain, groups, N_IPP, IPP_scale=1.0, units = 'dB'):
    '''Using pulse encoding schema, subgroup setup and coherent integration setup; convert from instantaneous gain to coherently integrated gain.
    
    :param float gain: Instantaneous gain, linear units or in dB.
    :param int groups: Number of subgroups from witch signals are coherently combined, assumes subgroups are identical.
    :param int N_IPP: Number of pulses to coherently integrate.
    :param float IPP_scale: Scale the IPP effective length in case e.g. the IPP is the same but the actual TX length is lowered.
    :param str units: If string equals 'dB', assume input and output units should be dB, else use linear scale.
    
    :return float: Gain after coherent integration, linear units or in dB.
    '''
    if units == 'dB':
        return gain + 10.0*np.log10( groups*N_IPP*IPP_scale )
    else:
        return gain*(groups*N_IPP*IPP_scale)


def coherrent_to_instantaneous(gain,groups,N_IPP,IPP_scale=1.0,units = 'dB'):
    '''Using pulse encoding schema, subgroup setup and coherent integration setup; convert from coherently integrated gain to instantaneous gain.
    
    :param float gain: Coherently integrated gain, linear units or in dB.
    :param int groups: Number of subgroups from witch signals are coherently combined, assumes subgroups are identical.
    :param int N_IPP: Number of pulses to coherently integrate.
    :param float IPP_scale: Scale the IPP effective length in case e.g. the IPP is the same but the actual TX length is lowered.
    :param str units: If string equals 'dB', assume input and output units should be dB, else use linear scale.
    
    :return float: Instantaneous gain, linear units or in dB.
    '''
    if units == 'dB':
        return gain - 10.0*np.log10( groups*N_IPP*IPP_scale )
    else:
        return gain/(groups*N_IPP*IPP_scale)
