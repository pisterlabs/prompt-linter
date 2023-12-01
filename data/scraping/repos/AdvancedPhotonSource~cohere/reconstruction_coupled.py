# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
cohere_core.reconstruction_coupled
=================================

This module controls a multipeak reconstruction process.
Refer to cohere_core-ui suite for use cases. The reconstruction can be started from GUI or using command line scripts, see :ref:`use`.
"""

import numpy as np
import os
import importlib
import cohere_core.controller.phasing as calc
import cohere_core.utilities.utils as ut
from multiprocessing import Process


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['reconstruction']


def set_lib(pkg, ndim=None):
    global devlib
    if pkg == 'cp':
        devlib = importlib.import_module('cohere_core.lib.cplib').cplib
    elif pkg == 'np':
        devlib = importlib.import_module('cohere_core.lib.nplib').nplib
    calc.set_lib(devlib)


def rec_process(lib, pars, peak_dirs, dev, continue_dir):
    set_lib(lib)
    # It is assumed that the calling script uses peak_dirs containing
    # peak orientation. It is parsed here, and passed to the CoupledRec constractor as list of
    # touples (<data directory>, <peak orientation>)
    peak_dir_orient = []
    packed_orientations = [(str(o[0]) + str(o[1]) + str(o[2])) for o in pars['orientations']]
    # find the directory that matches the packed orientation
    for dir in peak_dirs:
        found = False
        i = 0
        while i < len(packed_orientations) and not found:
            if dir.endswith(packed_orientations[i]):
                found = True
                peak_dir_orient.append((dir, pars['orientations'][i]))
            i += 1
    print(peak_dir_orient)

    worker = calc.CoupledRec(pars, peak_dir_orient)
    if worker.init_dev(dev[0]) < 0:
        return
    worker.init(continue_dir)
    if worker.iterate() < 0:
        return
    if 'save_dir' in pars:
        save_dir = pars['save_dir']
    else:
        save_dir = os.path.dirname(peak_dirs[0]) + '/results_phasing'
    worker.save_res(save_dir)


def reconstruction(lib, pars, peak_dirs, dev=None):
    """
    Controls multipeak reconstruction.

    This script is typically started with cohere-ui helper functions. It will start based on the configuration. The config file must have multipeak parameter set to True. In addition the config_mp with the peaks parameters must be included. The results will be saved in configured 'save_dir' parameter or in 'results_phasing' subdirectory if 'save_dir' is not defined.

    Parameters
    ----------
    lib : str
        library acronym to use for reconstruction. Supported:
        np - to use numpy,
        cp - to use cupy,
        torch - to use pytorch,

    pars : dict
        parameters reflecting configuration

    peak_dirs : list
        list of directories with data taken at each peak

    dev : int
        id defining GPU the this reconstruction will be utilizing

    """
    if 'init_guess' not in pars:
        pars['init_guess'] = 'random'
    if pars['init_guess'] == 'continue':
        continue_dir = pars['continue_dir']
    elif pars['init_guess'] == 'AI_guess':
        print('AI initial guess is not a valid choice for multi peak reconstruction')
        return -1
    else:
        continue_dir = None

    p = Process(target=rec_process, args=(lib, pars, peak_dirs, dev,
                                          continue_dir))
    p.start()
    p.join()
