# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Example script to plot the different CSR impedances

:Authors: **Markus Schwarz*

"""

import os

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import e as elCharge

from blond.beam.beam import Electron
from blond.impedances.impedance_sources import CoherentSynchrotronRadiation

mpl.use('Agg')

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

os.makedirs(this_directory + '../output_files/EX_22_fig/', exist_ok=True)


h, r_bend = 32e-3, 1.273  # chamber height [m], bending radius [m]

gamma = 40e6 / Electron().mass  # Lorentz factor

# frequencies at which to compute impedance (from 1e8 to 1e15 Hz)
freqs = 10**np.linspace(8, 15, num=200)

# approximate free-space CSR impedance increases as f^2/3
Z_fs_appr = CoherentSynchrotronRadiation(r_bend)
Z_fs_appr.imped_calc(freqs)

# The exact free-space CSR impedance increases as f^2/3 up to the critical frequency...
# ... and is exponentially suppressed above.
Z_fs = CoherentSynchrotronRadiation(r_bend, gamma=gamma)
Z_fs.imped_calc(freqs, high_frequency_transition=10)

f_crit = Z_fs.f_crit  # critical frequency [Hz]

# The approximate parallel-plates impedance is suppressed below the cut-off frequency...
# ... and approaches the approximate free-space CSR impedance for large frequencies.
Z_pp_appr = CoherentSynchrotronRadiation(r_bend, chamber_height=h)
Z_pp_appr.imped_calc(freqs, high_frequency_transition=10)

f_cut = Z_pp_appr.f_cut  # cut-off frequency [Hz]

# The parllel-plates impedance is supressed below the cut-off frequency, and approaches...
# ... the exact free-space impedance for larger frequencies
Z_pp = CoherentSynchrotronRadiation(r_bend, chamber_height=h, gamma=gamma)
Z_pp.imped_calc(freqs, low_frequency_transition=1e-4, high_frequency_transition=10)

plt.figure('re Z', clear=True)
plt.grid()
plt.xscale('log')
plt.xlabel('f / Hz')
plt.ylabel(r'Re$Z$ / k$\Omega$')
plt.ylim(-0.1, 12.1)
plt.plot(freqs, Z_fs.impedance.real / 1e3, '.-', label='free-space')
plt.plot(freqs, Z_pp.impedance.real / 1e3, label='parallel-plates')
plt.plot(freqs, Z_fs_appr.impedance.real / 1e3, label='free-space, approx.')
plt.plot(freqs, Z_pp_appr.impedance.real / 1e3, '--', label='parallel-plates, approx.')
plt.axvline(f_crit, color='black', linestyle='--', label=r'$f_{crit}$')
plt.axvline(f_cut, color='black', linestyle=':', label=r'$f_{cut}$')
plt.legend()
plt.tight_layout()
plt.savefig(this_directory + '../output_files/EX_22_fig/csr_impedance_real.png')

plt.figure('im Z', clear=True)
plt.grid()
plt.xscale('log')
plt.xlabel('f / Hz')
plt.ylabel(r'Re$Z$ / k$\Omega$')
plt.ylim(-10.1, 8.1)
plt.plot(freqs, Z_fs.impedance.imag / 1e3, '.-', label='free-space')
plt.plot(freqs, Z_pp.impedance.imag / 1e3, label='parallel-plates')
plt.plot(freqs, Z_fs_appr.impedance.imag / 1e3, label='free-space, approx.')
plt.plot(freqs, Z_pp_appr.impedance.imag / 1e3, '--', label='parallel-plates, approx.')
plt.axvline(f_crit, color='black', linestyle='--', label=r'$f_{crit}$')
plt.axvline(f_cut, color='black', linestyle=':', label=r'$f_{cut}$')
plt.legend()
plt.tight_layout()
plt.savefig(this_directory + '../output_files/EX_22_fig/csr_impedance_imag.png')

# compute the energy loss per turn as a cross-check
energy_loss = np.trapz(Z_fs.impedance.real, freqs) * elCharge**2  # [J]
energy_loss *= 2  # take negative frequencies into account
print(f'energy loss per turn (integrated spectrum): {energy_loss / elCharge:1.3f} eV')

# compare to textbook result
print(f"energy loss per turn (Sand's constant): {Electron().c_gamma * (40e6)**4 / r_bend:1.3f} eV")
