# -*- coding: utf-8 -*-

""" 
a script to test the com functions
"""

import numpy as np

from felpy.analysis.complex.coherence import get_longitudinal_coherence, get_longitudinal_coherence_new
from felpy.model.src.coherent import construct_SA1_pulse
from felpy.utils.os_utils import timing
from felpy.analysis.complex.coherence import get_complex_radial_profile

@timing 
def old(ii, dx):
    return get_longitudinal_coherence(ii, dx)

@timing 
def new(ii, dx):
    return get_longitudinal_coherence(ii, dx)
    
def speed_test(ii,dx):
    
    for i in range(5):
        
        ans1 = old(ii,dx)
        ans2 = new(ii,dx)
        print(ans1==ans2)
        print("")
        
    
if __name__ == '__main__':
    
    from scipy.signal import coherence
    from matplotlib import pyplot as plt
    
    nx, ny, nz = 500,500, 5
    ii = construct_SA1_pulse(nx, ny, nz, 5.0, 0.1)
    ii = ii.as_complex_array()
    #ii = get_complex_radial_profile(ii)[0]
    speed_test(1,ii)
    
    
    #c = coherence(ii, ii[:,-1:])[1]
    
