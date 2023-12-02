# +
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from nudd import config
from nudd.targets import nucleus_ar
from nudd.models import NeutrinoDipole
from nudd import coherent

from scipy.integrate import quad

# +
M2PERGEV = config.fm_conv * 1e15  # Conversion from m to /GeV
TON2GEV = 1e3 * config.c**2 / config.e0 / 1e9

E_MAX = 120  # Max energy to integrate to in keV

DETECTOR_CONFIGS = {'ESS10': {'threshold': 0.1,
                              'mass_det': 0.01,
                              'n_pot': 2.8e23,
                              'r': 0.3,
                              'baseline': 20,
                              'stat_sig': 0.05},

                    'CCM': {'threshold': 10,
                            'mass_det': 7,
                            'n_pot': 0.177e23,
                            'r': 0.0425,
                            'baseline': 20,
                            'stat_sig': 0.05},

                    'CNNS610': {'threshold': 15,
                                'mass_det': 0.61,
                                'n_pot': 1.5e23,
                                'r': 0.08,
                                'baseline': 28.4,
                                'stat_sig': 0.085},

                    'ESS': {'threshold': 20,
                            'mass_det': 1,
                            'n_pot': 2.8e23,
                            'r': 0.3,
                            'baseline': 20,
                            'stat_sig': 0.05}}


# -

class Detector:
    """Dataclass representing detector characteristics."""

    def __init__(self, nucleus, threshold, mass_det, n_pot, r, baseline, stat_sig):

        self.nucleus = nucleus
        self.mass_det = mass_det  # Detector mass in ton
        self.n_pot = n_pot
        self.r = r
        self.baseline = baseline  # in m
        self.threshold = threshold  # Threshold in keV
        self.stat_sig = stat_sig


    @property
    def eta(self):
        """The scaling for a detector."""

        mass_det = self.mass_det * TON2GEV
        baseline = self.baseline * M2PERGEV

        return mass_det * self.r * self.n_pot / (4 * np.pi * baseline**2)
    
    def quenching(self, E_Rs):
        """Return the quenching factor to relate the nuclear and electron equivalent energies (comes from 2104.03297 and 2003.10630): E_Rs[keVee]=Q_F*E_Rs[keVee]"""
        
        return 0.246 + 7.8e-4 * E_Rs * 10**6 # E_Rs must be in keV
    
    def efficiency(self, E_Rs):
        """Return the efficiency of the detector"""
        
        return 0.5 * (1 + np.tanh(self.quenching(E_Rs) * E_Rs * 10**6 - 4))

    def spectrum(self, E_Rs, nu):
        """Return the differential rate spectrum."""

        return self.nucleus.spectrum(np.asarray([E_Rs]), nu=nu) * self.eta
    
    def bin_edges(self, bin_width, E_MAX=120 / 1e6):
        """Return an array that contains bin edges (left and rigth values)"""
        
        return np.arange(self.threshold / 1e6, E_MAX+bin_width, bin_width) # The first element is the threshold energy
    
    def bin_spectrum(self, E_Rs, nu, bin_width, E_MAX= 120 / 1e6, int_method = 'quad'):
        """Return the bined integrated rate spectrum and the bined energy"""
        
        bin_edges = self.bin_edges(bin_width, E_MAX= 120 /1e6) # This generates the bins
        z         = np.zeros(np.size(bin_edges)-1)
        
        if int_method == 'trapz':
            
            trapz_sampling = 100
            for i in np.arange(np.size(bin_edges)-1): # Make ONE integration per bin (extreme right value is therefore not included)
                #x=E_Rs[np.where((E_Rs>=bin_edges[i])*(E_Rs<=bin_edges[i+1]))] # Select the energies inside the bin
                #y=spectrum[np.where((E_Rs>=bin_edges[i])*(E_Rs<=bin_edges[i+1]))] * self.efficiency(x) # Select the differential rate values inside the bin
                
                E_Rs_bin = np.geomspace(bin_edges[i], bin_edges[i+1], trapz_sampling)
                integrand = self.spectrum(E_Rs_bin, nu) * self.efficiency(E_Rs_bin)

                z[i]=np.trapz(integrand, E_Rs_bin) # Integrate in the bin i (probably there are better ways to integrate haha)
        if int_method == 'quad':
            
            func = lambda x: detector_sm.spectrum(x, nu) * self.efficiency(x)
        
            for i in np.arange(np.size(bin_edges) - 1): # Make ONE integration per bin (extreme right value is therefore not included)
                nu = None
                z[i] = quad(func, bin_edges[i], bin_edges[i+1],)[0] # Integrate in the bin i 
        
        return bin_edges, z 

E_Rs = np.geomspace(1e-1, E_MAX, 1000) / 1e6  # In GEV!!

# # Let's see the differece between integrating with quad and trapz

# +
detector_sm = Detector(nucleus_ar, **DETECTOR_CONFIGS['ESS'])

bin_edges, z = detector_sm.bin_spectrum(E_Rs, None, 10 / 1e6, E_MAX / 1e6, int_method = 'trapz')
_, z_q       = detector_sm.bin_spectrum(E_Rs, None, 10 / 1e6, E_MAX / 1e6, int_method = 'quad') 

bin_width   = (bin_edges[1] - bin_edges[0]) / 2
bin_centers = bin_edges[:-1] + bin_width

fig, ax = plt.subplots(2,1, sharex = True, gridspec_kw = {'hspace':0})

ax[0].scatter(bin_centers * 1e6, z, label = 'trapz', color='black')
ax[0].scatter(bin_centers * 1e6, z_q, label = 'quad', color='red', marker = '+')
ax[0].set_ylabel('#  Events')
ax[0].legend()

ax[1].plot(bin_centers * 1e6, np.abs(z - z_q) / z_q)
ax[1].set_ylabel(r'$|t - q| / q$')
ax[1].set_xlabel('[GeV]')
# -

# # Let's make a quick profiling of both methods

# +
from time import time
niter = 100

start = time()
for i in range(niter):
    bin_edges, z_q = detector_sm.bin_spectrum(E_Rs, None, 10 / 1e6, E_MAX / 1e6, int_method = 'quad') 
end = time()
print(f'It took {end - start} seconds with quad!')

start = time()
for i in range(niter):
    bin_edges, z_q = detector_sm.bin_spectrum(E_Rs, None, 10 / 1e6, E_MAX / 1e6, int_method = 'trapz') 
end = time()
print(f'It took {end - start} seconds with trapz!')
# -


