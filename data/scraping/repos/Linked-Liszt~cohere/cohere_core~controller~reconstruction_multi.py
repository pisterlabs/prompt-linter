# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
cohere_core.reconstruction_multi
================================

This module controls a multiple reconstructions.
Refer to cohere_core-ui suite for use cases. The reconstruction can be started from GUI x or using command line scripts x.
"""

import os
import numpy as np
import importlib
import cohere_core.utilities.utils as ut
import cohere_core.controller.phasing as calc
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Process
from functools import partial
import threading


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['multi_rec',
           'reconstruction']


class Devices:
    def __init__(self, devices):
        self.devices = devices
        self.index = 0

    def assign_gpu(self):
        thr = threading.current_thread()
        thr.gpu = self.devices[self.index]
        self.index = self.index + 1


def set_lib(pkg, ndim=None):
    global devlib
    if pkg == 'af':
        if ndim == 1:
            devlib = importlib.import_module('cohere_core.lib.aflib').aflib1
        elif ndim == 2:
            devlib = importlib.import_module('cohere_core.lib.aflib').aflib2
        elif ndim == 3:
            devlib = importlib.import_module('cohere_core.lib.aflib').aflib3
        else:
            raise NotImplementedError
    elif pkg == 'cp':
        devlib = importlib.import_module('cohere_core.lib.cplib').cplib
    elif pkg == 'np':
        devlib = importlib.import_module('cohere_core.lib.nplib').nplib
    calc.set_lib(devlib, pkg=='af')


def single_rec_process(metric_type, gen, rec_attrs):
    """
    This function runs a single reconstruction process.

    Parameters
    ----------
    proc : str
        string defining library used 'cpu' or 'opencl' or 'cuda'

    pars : Object
        Params object containing parsed configuration

    data : numpy array
        data array

    req_metric : str
        defines metric that will be used if GA is utilized

    dirs : list
        tuple of two elements: directory that contain results of previous run or None, and directory where the results of this processing will be saved

    Returns
    -------
    metric : float
        a calculated characteristic of the image array defined by the metric
    """
    worker, prev_dir, save_dir = rec_attrs
    thr = threading.current_thread()
    if worker.init_dev(thr.gpu) < 0:
        metric = None
    else:
        worker.init(prev_dir, gen)

        if gen is not None and gen > 0:
            worker.breed()

        ret_code = worker.iterate()
        if ret_code == 0:
            worker.save_res(save_dir)
            metric = worker.get_metric(metric_type)
        else:    # bad reconstruction
            metric = None
    return [metric, save_dir]


def multi_rec(lib, save_dir, devices, no_recs, pars, datafile, prev_dirs, metric_type='chi', gen=None, q=None):
    """
    This function controls the multiple reconstructions.

    Parameters
    ----------
    lib : str
        library acronym to use for reconstruction. Supported:
        np - to use numpy
        cp - to use cupy
        af - to use arrayfire
        cpu, opencl, or cuda - to use specified library of arrayfire

    save_dir : str
        a directory where the subdirectories will be created to save all the results for multiple reconstructions

    devices : list
        list of GPUs available for this reconstructions

    no_recs : int
        number of reconstructions

    pars : dict
        parameters for reconstruction

    datafie : str
        name of file containing data for reconstruction

    previous_dirs : list
        directories that hols results of previous reconstructions if it is continuation or None(s)

    metric_type : str
        a metric defining algorithm by which to evaluate the quality of reconstructed array

    gen : int
        which generation is the reconstruction for

    q : queue
        if provided the results will be queued

    Returns
    -------
    None
    """
    # evals = []
    # prev_dir_seq =[]
    results = []

    def collect_result(result):
        results.append(result)

    if lib == 'af' or lib == 'cpu' or lib == 'opencl' or lib == 'cuda':
        if datafile.endswith('tif') or datafile.endswith('tiff'):
            try:
                data = ut.read_tif(datafile)
            except:
                print ('could not load data file', datafile)
                return
        elif datafile.endswith('npy'):
            try:
                data = np.load(datafile)
            except:
                print ('could not load data file', datafile)
                return
        else:
            print ('no data file found')
            return
        print('data shape', data.shape)
        set_lib('af', len(data.shape))
        if lib != 'af':
            devlib.set_backend(lib)
    else:
        set_lib(lib)

    workers = [calc.Rec(pars, datafile) for _ in range(no_recs)]
    dev_obj = Devices(devices)
    iterable = []
    save_dirs = []

    for i in range(len(workers)):
        save_sub = save_dir + '/' + str(i)
        save_dirs.append(save_sub)
        iterable.append((workers[i], prev_dirs[i], save_sub))
    func = partial(single_rec_process, metric_type, gen)
    with Pool(processes=len(devices), initializer=dev_obj.assign_gpu, initargs=()) as pool:
        pool.map_async(func, iterable, callback=collect_result)
        pool.close()
        pool.join()
        pool.terminate()

    if q is not None:
        q.put(results[0])


def reconstruction(lib, conf_file, datafile, dir, devices):
    """
    Controls multiple reconstructions, the reconstructions run concurrently.

    This script is typically started with cohere_core-ui helper functions. The 'init_guess' parameter in the configuration file defines whether guesses are random, or start from some saved states. It will set the initial guesses accordingly and start phasing process, running each reconstruction in separate thread. The results will be saved in configured 'save_dir' parameter or in 'results_phasing' subdirectory if 'save_dir' is not defined.

    Parameters
    ----------
    lib : str
        library acronym to use for reconstruction. Supported:
        np - to use numpy,
        cp - to use cupy,
        af - to use arrayfire,
        cpu, opencl, or cuda - to use specified library of arrayfire

    conf_file : str
        configuration file name

    datafile : str
        data file name

    dir : str
        a parent directory that holds the reconstructions. For example experiment directory or scan directory.

    devices : list
        list of GPUs available for this reconstructions

    """
    pars = ut.read_config(conf_file)

    if 'reconstructions' in pars:
        reconstructions = pars['reconstructions']
    else:
        reconstructions = 1

    prev_dirs = []
    if 'init_guess' not in pars:
        pars['init_guess'] = 'random'
    if pars['init_guess'] == 'continue':
        continue_dir = pars['continue_dir']
        for sub in os.listdir(continue_dir):
            image, support, coh = ut.read_results(continue_dir + '/' + sub + '/')
            if image is not None:
                prev_dirs.append(continue_dir + '/' + sub)
        if len(prev_dirs) < reconstructions:
            prev_dirs = prev_dirs + (reconstructions - len(prev_dirs)) * [None]
    elif pars['init_guess'] == 'AI_guess':
        print('multiple reconstruction do not support AI_guess initial guess')
        return
    else:
        for _ in range(reconstructions):
            prev_dirs.append(None)
    if 'save_dir' in pars:
        save_dir = pars['save_dir']
    else:
        filename = conf_file.split('/')[-1]
        save_dir = dir + '/' + filename.replace('config_rec', 'results_phasing')

    p = Process(target=multi_rec, args=(lib, save_dir, devices, reconstructions, pars, datafile, prev_dirs))
    p.start()
    p.join()
