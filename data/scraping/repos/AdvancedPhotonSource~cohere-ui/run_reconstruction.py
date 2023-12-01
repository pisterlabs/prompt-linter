# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This user script manages reconstruction(s).
Depending on configuration it starts either single reconstruction, GA, or multiple reconstructions. In multiple reconstruction scenario or split scans the script runs concurrent reconstructions.
"""

__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['rec_process',
           'get_gpu_use',
           'manage_reconstruction',
           'main']

import sys
import signal
import os
import argparse
from multiprocessing import Process, Queue
import cohere_core as cohere
import util.util as ut
import convertconfig as conv

MEM_FACTOR = 170
GA_MEM_FACTOR = 250
ADJUST = 0.0


def rec_process(proc, conf_file, datafile, dir, gpus, r, q):
    """
    Calls the reconstruction function in a module identified by parameter. After the reconstruction is finished, it enqueues th eprocess id wit associated list of gpus.
    Parameters
    ----------
    proc : str
        processing library, chices are cpu, cuda, opencl
    conf_file : str
        configuration file with reconstruction parameters
    datafile : str
        name of file containing data
    dir : str
        parent directory to the <prefix>/results, or results directory
    gpus : list
       a list of gpus that will be used for reconstruction
    r : str
       a string indentifying the module to use for reconstruction
    q : Queue
       a queue that returns tuple of procees id and associated gpu list after the reconstruction process is done
    Returns
    -------
    nothing
    """
    if r == 'g':
        cohere.reconstruction_GA.reconstruction(proc, conf_file, datafile, dir, gpus)
    elif r == 'm':
        cohere.reconstruction_multi.reconstruction(proc, conf_file, datafile, dir, gpus)
    elif r == 's':
        cohere.reconstruction_single.reconstruction(proc, conf_file, datafile, dir, gpus)
    q.put((os.getpid(), gpus))


def get_gpu_use(devices, no_dir, no_rec, data_shape, pc_in_use, ga_in_use):
    """
    Determines the use case, available GPUs that match configured devices, and selects the optimal distribution of reconstructions on available devices.
    Parameters
    ----------
    devices : list
        list of configured GPU ids to use for reconstructions. If -1, operating system is assigning it
    no_dir : int
        number of directories to run independent reconstructions
    no_rec : int
        configured number of reconstructions to run in each directory
    data_shape : tuple
        shape of data array, needed to estimate how many reconstructions will fit into available memory
    pc_in_use : boolean
        True if partial coherence is configured
    Returns
    -------
    gpu_use : list
        a list of int indicating number of runs per consecuitive GPUs
    """
    from functools import reduce

    if sys.platform == 'darwin':
        # the gpu library is not working on OSX, so run one reconstruction on each GPU
        gpu_load = len(devices) * [1, ]
    else:
        # find size of data array
        data_size = reduce((lambda x, y: x * y), data_shape) / 1000000.
        mem_factor = MEM_FACTOR
        if ga_in_use:
            mem_factor = GA_MEM_FACTOR
        rec_mem_size = data_size * mem_factor
        if pc_in_use:
            rec_mem_size = rec_mem_size * 2
        gpu_load = ut.get_gpu_load(rec_mem_size, devices)

    no_runs = no_dir * no_rec
    gpu_distribution = ut.get_gpu_distribution(no_runs, gpu_load)
    gpu_use = []
    available = reduce((lambda x, y: x + y), gpu_distribution)
    dev_index = 0
    i = 0
    while i < available:
        if gpu_distribution[dev_index] > 0:
            gpu_use.append(devices[dev_index])
            gpu_distribution[dev_index] = gpu_distribution[dev_index] - 1
            i += 1
        dev_index += 1
        dev_index = dev_index % len(devices)
    if no_dir > 1:
        gpu_use = [gpu_use[x:x + no_rec] for x in range(0, len(gpu_use), no_rec)]

    return gpu_use


def manage_reconstruction(experiment_dir, rec_id=None):
    """
    This function starts the interruption discovery process and continues the recontruction processing.
    It reads configuration file defined as <experiment_dir>/conf/config_rec.
    If multiple generations are configured, or separate scans are discovered, it will start concurrent reconstructions.
    It creates image.npy file for each successful reconstruction.
    Parameters
    ----------
    experiment_dir : str
        directory where the experiment files are loacted
    rec_id : str
        optional, if given, alternate configuration file will be used for reconstruction, (i.e. <rec_id>_config_rec)
    Returns
    -------
    nothing
    """
    print('starting reconstruction')
    experiment_dir = experiment_dir.replace(os.sep, '/')
    # the rec_id is a postfix added to config_rec configuration file. If defined, use this configuration.
    conf_dir = experiment_dir + '/conf'
    # convert configuration files if needed
    main_conf = conf_dir + '/config'
    if os.path.isfile(main_conf):
        main_config_map = ut.read_config(main_conf)
        if main_config_map is None:
            print ("info: can't read " + main_conf + " configuration file")
            return None
    else:
        print("info: missing " + main_conf + " configuration file")
        return None

    if 'converter_ver' not in main_config_map or conv.get_version() is None or conv.get_version() < main_config_map['converter_ver']:
        main_config_map = conv.convert(conf_dir, 'config')
    # verify main config file
    er_msg = cohere.verify('config', main_config_map)
    if len(er_msg) > 0:
        # the error message is printed in verifier
        return None

    if rec_id is None:
        conf_file = conf_dir + '/config_rec'
    else:
        conf_file = conf_dir + '/config_rec_' + rec_id

    rec_config_map = ut.read_config(conf_file)
    if rec_config_map is None:
        return

    # verify configuration
    er_msg = cohere.verify('config_rec', rec_config_map)
    if len(er_msg) > 0:
        # the error message is printed in verifier
        return None

    # find which library to run it on, default is numpy ('np')
    if 'processing' in rec_config_map:
        proc = rec_config_map['processing']
    else:
        proc = 'auto'

    lib = 'np'
    if proc == 'auto':
        try:
            import cupy
            lib = 'cp'
        except:
            try:
                import torch
                lib = 'torch'
            except:
               pass
    elif proc == 'cp':
        try:
            import cupy
            lib = 'cp'
        except:
            print('cupy is not installed, select different library (proc)')
            return
    elif proc == 'torch':
        try:
            import torch
            lib = 'torch'
        except:
            print('pytorch is not installed, select different library (proc)')
            return
    elif proc == 'np':
        pass  # lib set to 'np'
    else:
        print('invalid "proc" value', proc, 'is not supported')
        return

    # for multipeak reconstruction divert here
    if 'multipeak' in main_config_map and main_config_map['multipeak']:
        config_map = ut.read_config(experiment_dir + "/conf/config_mp")
        config_map.update(main_config_map)
        config_map.update(rec_config_map)
        if 'device' in config_map:
            dev = config_map['device']
        else:
            dev = [-1]
        peak_dirs = []
        for dir in os.listdir(experiment_dir):
            if dir.startswith('mp'):
                peak_dirs.append(experiment_dir + '/' + dir)
        cohere.reconstruction_coupled.reconstruction(lib, config_map, peak_dirs, dev)
    else:
        # exp_dirs_data list hold pairs of data and directory, where the directory is the root of data/data.tif file, and
        # data is the data.tif file in this directory.
        exp_dirs_data = []
        # experiment may be multi-scan in which case reconstruction will run for each scan
        for dir in os.listdir(experiment_dir):
            if dir.startswith('scan') or dir.startswith('mp'):
                datafile = experiment_dir + '/' + dir + '/phasing_data/data.tif'
                if os.path.isfile(datafile):
                    exp_dirs_data.append((datafile, experiment_dir + '/' + dir))
        # if there are no scan directories, assume it is combined scans experiment
        if len(exp_dirs_data) == 0:
            # in typical scenario data_dir is not configured, and it is defaulted to <experiment_dir>/data
            # the data_dir is ignored in multi-scan scenario
            if 'data_dir' in rec_config_map:
                data_dir = rec_config_map['data_dir'].replace(os.sep, '/')
            else:
                data_dir = experiment_dir + '/phasing_data'
            datafile = data_dir + '/data.tif'
            if os.path.isfile(datafile):
                exp_dirs_data.append((datafile, experiment_dir))
        no_runs = len(exp_dirs_data)
        if no_runs == 0:
            print('did not find data.tif file(s). ')
            return
        if 'ga_generations' in rec_config_map:
            generations = rec_config_map['ga_generations']
        else:
            generations = 0
        if 'reconstructions' in rec_config_map:
            reconstructions = rec_config_map['reconstructions']
        else:
            reconstructions = 1
        device_use = []
        if lib == 'np':
            cpu_use = [-1] * reconstructions
            if no_runs > 1:
                for _ in range(no_runs):
                    device_use.append(cpu_use)
            else:
                device_use = cpu_use
        else:
            if 'device' in rec_config_map:
                devices = rec_config_map['device']
            else:
                devices = [-1]

            if no_runs * reconstructions > 1:
                data_shape = cohere.read_tif(exp_dirs_data[0][0]).shape
                device_use = get_gpu_use(devices, no_runs, reconstructions, data_shape, 'pc' in rec_config_map['algorithm_sequence'], generations > 1)
            else:
                device_use = devices

        if no_runs == 1:
            if len(device_use) == 0:
                device_use = [-1]
            dir_data = exp_dirs_data[0]
            datafile = dir_data[0]
            dir = dir_data[1]
            if generations > 1:
                cohere.reconstruction_GA.reconstruction(lib, conf_file, datafile, dir, device_use)
            elif reconstructions > 1:
                cohere.reconstruction_multi.reconstruction(lib, conf_file, datafile, dir, device_use)
            else:
                cohere.reconstruction_single.reconstruction(lib, conf_file, datafile, dir, device_use)
        else:
            if len(device_use) == 0:
                device_use = [[-1]]
            else:
                # check if is it worth to use last chunk
                if lib != 'np' and len(device_use[0]) > len(device_use[-1]) * 2:
                    device_use = device_use[0:-1]
            if generations > 1:
                r = 'g'
            elif reconstructions > 1:
                r = 'm'
            else:
                r = 's'
            q = Queue()
            for gpus in device_use:
                q.put((None, gpus))
            # index keeps track of the multiple directories
            index = 0
            processes = {}
            pr = []
            while index < no_runs:
                pid, gpus = q.get()
                if pid is not None:
                    os.kill(pid, signal.SIGKILL)
                    del processes[pid]
                datafile = exp_dirs_data[index][0]
                dir = exp_dirs_data[index][1]
                p = Process(target=rec_process, args=(lib, conf_file, datafile, dir, gpus, r, q))
                p.start()
                pr.append(p)
                processes[p.pid] = index
                index += 1

            for p in pr:
                p.join()

            # close the queue
            q.close()

        print('finished reconstruction')


def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="experiment directory.")
    parser.add_argument("--rec_id", help="reconstruction id, a postfix to 'results_phasing_' directory")
    args = parser.parse_args()
    experiment_dir = args.experiment_dir

    if args.rec_id:
        manage_reconstruction(experiment_dir, args.rec_id)
    else:
        manage_reconstruction(experiment_dir)


if __name__ == "__main__":
    main(sys.argv[1:])

