# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This user script processes reconstructed image for visualization.

After the script is executed the experiment directory will contain image.vts file for each reconstructed image in the given directory tree.
"""

__author__ = "Ross Harder"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['process_dir',
           'get_conf_dict',
           'handle_visualization',
           'main']

import argparse
import sys
import os
import numpy as np
from functools import partial
from multiprocessing import Pool, cpu_count
import importlib
import convertconfig as conv
import cohere_core as cohere
import util.util as ut
from uvw import DataArray, StructuredGrid


def arrays_to_vts(arrays, filename, coords, shape):
    """
    Writes arrays into the vts file.

    Parameters
    ----------
    arrays : dict
        {array_name : array}, the arrays must have the same shape
    filename : str
        name of the vts file the arrays will be written to
    coords : ndarray
        2D numpy array of point coordinates
    shape : list
        shape of arrays

    """
    grid = StructuredGrid(filename, coords, shape)
    for arr_name in arrays.keys():
        grid.addPointData(DataArray(arrays[arr_name], list(range(len(shape)))[::-1], arr_name))
    grid.write()


def get_coordinates3(shape, Tdir):
    """
    Creates grid and applies geometry to get coordinates.

    Parameters
    ----------
    shape : tuple
        shape of reconstructed array
    Tdir : ndarray
        geometry in direct space

    Returns
    -------
    coords : ndarray
        2D numpy array of point coordinates

    """
    dims = list(shape)
    dxdir = 1.0 / shape[0]
    dydir = 1.0 / shape[1]
    dzdir = 1.0 / shape[2]

    r = np.mgrid[0:dims[0] * dxdir:dxdir, 0:dims[1] * dydir:dydir, 0:dims[2] * dzdir:dzdir]

    r.shape = 3, dims[0] * dims[1] * dims[2]

    coords = np.dot(Tdir, r).transpose()

    return coords


def process_dir(coords, rampups, new_shape, make_twin, res_dir):
    """
    Loads arrays from files in results directory, crops to the new size, and writes to vts files.

    Parameters
    ----------
    coords : ndarray
        2D numpy array of point coordinates
    rampups : int
        a number by which to increase size of array when doing interpolation
    new_shape : list
        size of the array after applying crop
    make_twin : boolean
        if True, save twin image and support
    res_dir : str
        directory where to read the image.npy/support.npy results of phasing

    Returns
    -------
    nothing
    """
    save_dir = res_dir.replace('_phasing', '_viz')
    # create dir if does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # image file was checked in calling function
    imagefile = res_dir + '/image.npy'
    try:
        image = np.load(imagefile)
        ut.save_tif(image, save_dir + '/image.tif')
    except:
        print('cannot load file', imagefile)
        return

    support = None
    coh = None

    supportfile = res_dir + '/support.npy'
    if os.path.isfile(supportfile):
        try:
            support = np.load(supportfile)
            ut.save_tif(support, save_dir + '/support.tif')
        except:
            print('cannot load file', supportfile)
    else:
        print('support file is missing in ' + res_dir + ' directory')

    cohfile = res_dir + '/coherence.npy'
    if os.path.isfile(cohfile):
        try:
            coh = np.load(cohfile)
        except:
            print('cannot load file', cohfile)

    image, support = ut.center(image, support)
    if rampups > 1:
        image = ut.remove_ramp(image, ups=rampups)

    # save image
    image = ut.crop_center(image, new_shape)
    arrays = {"imAmp": np.abs(image), "imPh": np.angle(image)}
    arrays_to_vts(arrays, save_dir + '/image.vts', coords, new_shape)

    # save support
    if support is not None:
        support = ut.crop_center(support, new_shape)
        arrays = {'support': support}
        arrays_to_vts(arrays, save_dir + '/support.vts', coords, new_shape)

    if coh is not None:
        coh = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(coh)))
        coh = ut.get_zero_padded_centered(coh, new_shape)
        arrays = {"cohAmp": np.abs(coh), "cohPh": np.angle(coh)}
        arrays_to_vts(arrays, save_dir + '/coherence.vts', coords, new_shape)

    if make_twin:
        image = np.conjugate(np.flip(image))
        if support is not None:
            support = np.flip(support)
        image, support = ut.center(image, support)
        arrays = {"imAmp": abs(image), "imPh": np.angle(image)}
        arrays_to_vts(arrays, save_dir + '/twin_image.vts', coords, new_shape)
        if support is not None:
            arrays = {'support': support}
            arrays_to_vts(arrays, save_dir + '/twin_support.vts', coords, new_shape)


def get_conf_dict(experiment_dir):
    """
    Reads configuration files and creates dictionary with parameters that are needed for visualization.

    Parameters
    ----------
    experiment_dir : str
        directory where the experiment files are located

    Returns
    -------
    conf_dict : dict
        a dictionary containing configuration parameters
    """
    experiment_dir = experiment_dir.replace(os.sep, '/')
    if not os.path.isdir(experiment_dir):
        print("Please provide a valid experiment directory")
        return None
    conf_dir = experiment_dir + '/conf'

    main_conf_file = conf_dir + '/config'
    main_conf_map = ut.read_config(main_conf_file)
    if main_conf_map is None:
        return None

    # convert configuration files if needed
    if 'converter_ver' not in main_conf_map or conv.get_version() is None or conv.get_version() < main_conf_map[
        'converter_ver']:
        conv.convert(conf_dir)
        # re-parse config
        main_conf_map = ut.read_config(main_conf_file)

    er_msg = cohere.verify('config', main_conf_map)
    if len(er_msg) > 0:
        # the error message is printed in verifier
        return None

    disp_conf = conf_dir + '/config_disp'

    # parse the conf once here and save it in dictionary, it will apply to all images in the directory tree
    conf_dict = ut.read_config(disp_conf)
    if conf_dict is None:
        return None
    er_msg = cohere.verify('config_disp', conf_dict)
    if len(er_msg) > 0:
        # the error message is printed in verifier
        return None

    if 'beamline' in main_conf_map:
        conf_dict['beamline'] = main_conf_map['beamline']
    else:
        print('Beamline must be configured in configuration file ' + main_conf_file)
        return None

    # get specfile and last_scan from the config file and add it to conf_dict
    if 'specfile' in main_conf_map and 'scan' in main_conf_map:
        conf_dict['specfile'] = main_conf_map['specfile'].replace(os.sep, '/')
        scan = main_conf_map['scan']
        last_scan = scan.split(',')[-1].split('-')[-1]
        conf_dict['last_scan'] = int(last_scan)
    else:
        print("specfile or scan range not in main config")

    # get binning from the config_data file and add it to conf_dict
    data_conf = conf_dir + '/config_data'
    data_conf_map = ut.read_config(data_conf)
    if data_conf_map is None:
        return conf_dict
    if 'binning' in data_conf_map:
        conf_dict['binning'] = data_conf_map['binning']
    if 'separate_scans' in data_conf_map and data_conf['separate_scans'] or 'separate_scan_ranges' in data_conf_map and  data_conf['separate_scan_ranges']:
        conf_dict['separate'] = True
    else:
        conf_dict['separate'] = False

    return conf_dict


def handle_visualization(experiment_dir, rec_id=None, image_file=None):
    """
    3D visualization in direct space.

    If the image_file parameter is defined, the file is processed and vts file saved. Otherwise this function determines root directory with results that should be processed for visualization. Multiple images will be processed concurrently.

    Parameters
    ----------
    experiment_dir : str
        experiment directory
    rec_id : str
        optional, a string identifying alternate reconstruction
    image_file : str
        optional, if given, only this file is visualized

    Returns
    -------
    nothing
    """
    experiment_dir = experiment_dir.replace(os.sep, '/')
    print ('starting visualization process')
    conf_dict = get_conf_dict(experiment_dir)
    if conf_dict is None:
        return

    try:
        disp = importlib.import_module('beamlines.' + conf_dict['beamline'] + '.disp')
    except:
        print ('cannot import beamlines.' + conf_dict['beamline'] + '.disp module.')
        return

    try:
        params = disp.DispalyParams(conf_dict)
    except Exception as e:
        print ('exception', e)
        return

    det_obj = None
    diff_obj = None
    try:
        detector_name = params.detector
        print()
        try:
            det = importlib.import_module('beamlines.' + conf_dict['beamline'] + '.detectors')
            try:
                det_obj = det.create_detector(detector_name)
            except:
                print('detector', detector_name, 'is not defined in beamlines detectors')
        except:
            print('problem importing detectors file from beamline module')
    except:
        pass
    try:
        diffractometer_name = params.diffractometer
        try:
            diff = importlib.import_module('beamlines.' + conf_dict['beamline'] + '.diffractometers')
            try:
                diff_obj = diff.create_diffractometer(diffractometer_name)
            except:
                print ('diffractometer', diffractometer_name, 'is not defined in beamlines detectors')
        except:
             print('problem importing diffractometers file from beamline module')
    except:
        pass

    if not params.set_instruments(det_obj, diff_obj):
        return

    try:
        rampups = params.rampsup
    except:
        rampups = 1

    if 'make_twin' in conf_dict:
        make_twin = conf_dict['make_twin']
    else:
        make_twin = False

    dirs = []
    if image_file is not None:
        image_file = image_file.replace(os.sep, '/')
        if os.path.isfile(image_file):
            dirs.append(os.path.dirname(image_file).replace(os.sep, '/'))
        else:
            print(image_file, 'file is missing')
            return
    else:
        if 'results_dir' in conf_dict:
            results_dir = conf_dict['results_dir'].replace(os.sep, '/')
        elif conf_dict['separate']:
            results_dir = experiment_dir
        elif rec_id is not None:
            results_dir = experiment_dir + '/results_phasing_' + rec_id
        else:
            results_dir = experiment_dir + '/results_phasing'

        # find directories with image.npy file in the root of results_dir
        for (dirpath, dirnames, filenames) in os.walk(results_dir):
            for file in filenames:
                if file.endswith('image.npy'):
                    dirs.append((dirpath).replace(os.sep, '/'))

    if len(dirs) == 0:
        print ('no image.npy files found in the directory tree', results_dir)
        return
    else:
        # find shape without loading the array
        with open(dirs[0] + '/image.npy', 'rb') as f:
            np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        # geometry depends on beamline instruments, get it from the beamline object
        (Trecip, Tdir) = disp.get_geometry3(shape, params)
        # find coordinates
        new_shape = [int(shape[i] * params.crop[i]) for i in range(len(shape))]
        coords = get_coordinates3(new_shape, Tdir)

    if len(dirs) == 1:
        process_dir(coords, rampups, new_shape, make_twin, dirs[0])
    elif len(dirs) >1:
        func = partial(process_dir, coords, rampups, new_shape, make_twin)
        no_proc = min(cpu_count(), len(dirs))
        with Pool(processes = no_proc) as pool:
           pool.map_async(func, dirs)
           pool.close()
           pool.join()
    print ('done with processing display')


def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="experiment directory")
    parser.add_argument("--image_file", help="a file in .npy format to be processed for visualization")
    parser.add_argument("--rec_id", help="alternate reconstruction id")
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    rec_id = args.rec_id
    if args.image_file:
        handle_visualization(experiment_dir, args.rec_id, args.image_file)
    else:
        handle_visualization(experiment_dir, args.rec_id)


if __name__ == "__main__":
    main(sys.argv[1:])

# python run_disp.py experiment_dir
