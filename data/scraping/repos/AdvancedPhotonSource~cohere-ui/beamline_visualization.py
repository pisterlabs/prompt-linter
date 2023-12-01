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
           'get_conf_dicts',
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
from tvtk.api import tvtk
import multipeak as mp


class CXDViz:
    """
    CXDViz(self, crop, geometry)
    ===================================
    Class, generates files for visualization from reconstructed suite.
    crop : list
        list of fractions; the fractions will be multipled by dimensions to derive region to visualize
    geometry : tuple of arrays
        arrays containing geometry in reciprocal and direct space
    """
    __all__ = ['visualize']

    dir_arrs = {}
    recip_arrs = {}

    def __init__(self, crop, geometry):
        """
        The constructor creates objects assisting with visualization.
        Parameters
        ----------
        crop : tuple or list
            list of fractions; the fractions will be applied to each dimension to derive region to visualize
        geometry : tuple of arrays
            arrays containing geometry in reciprocal and direct space
        Returns
        -------
        constructed object
        """
        self.crop = crop
        self.Trecip, self.Tdir = geometry
        self.dirspace_uptodate = 0
        self.recipspace_uptodate = 0


    def visualize(self, image, support, coh, save_dir, unwrap=False, is_twin=False):
        """
        Manages visualization process. Saves the results in a given directory in files: image.vts, support.vts, and coherence.vts. If is_twin then the saved files have twin prefix.
        Parameters
        ----------
        image : ndarray
            image array
        support : ndarray
            support array or None
        coh : ndarray
            coherence array or None
        save_dir : str
            a directory to save the results
        is_twin : boolean
            True if the image array is result of reconstruction, False if is_twin of reconstructed array.
        """
        save_dir = save_dir.replace(os.sep, '/')
        arrays = {"imAmp": abs(image), "imPh": np.angle(image)}

          # unwrap phase here
        if unwrap:
            from skimage import restoration
            arrays['imUwPh'] = restoration.unwrap_phase(arrays['imPh'])

        self.add_ds_arrays(arrays)
        if is_twin:
            self.write_directspace(save_dir + '/twin_image')
        else:
            self.write_directspace(save_dir + '/image')
        self.clear_direct_arrays()
        if support is not None:
            arrays = {"support": support}
            self.add_ds_arrays(arrays)
            if is_twin:
                self.write_directspace(save_dir + '/twin_support')
            else:
                self.write_directspace(save_dir + '/support')
            self.clear_direct_arrays()

        if coh is not None:
            coh = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(coh)))
            coh = ut.get_zero_padded_centered(coh, image.shape)
            arrays = {"cohAmp": np.abs(coh), "cohPh": np.angle(coh)}
            self.add_ds_arrays(arrays)
            self.write_directspace(save_dir + '/coherence')
            self.clear_direct_arrays()


    def update_dirspace(self, shape, orig_shape):
        """
        Updates direct space grid.
        Parameters
        ----------
        shape : tuple
            shape of reconstructed array
        Returns
        -------
        nothing
        """
        dims = list(shape)
        self.dxdir = 1.0 / orig_shape[0]
        self.dydir = 1.0 / orig_shape[1]
        self.dzdir = 1.0 / orig_shape[2]

        r = np.mgrid[
            0:dims[0] * self.dxdir:self.dxdir, \
            0:dims[1] * self.dydir:self.dydir, \
            0:dims[2] * self.dzdir:self.dzdir]

        r.shape = 3, dims[0] * dims[1] * dims[2]

        self.dir_coords = np.dot(self.Tdir, r).transpose()

        self.dirspace_uptodate = 1

    def update_recipspace(self, shape):
        """
        Updates reciprocal space grid.
        Parameters
        ----------
        shape : tuple
            shape of reconstructed array
        Returns
        -------
        nothing
        """
        dims = list(shape)
        q = np.mgrid[0:dims[0], 0:dims[1], 0:dims[2]]

        q.shape = 3, dims[0] * dims[1] * dims[2]

        self.recip_coords = np.dot(self.Trecip, q).transpose()
        self.recipspace_uptodate = 1


    def clear_direct_arrays(self):
        self.dir_arrs.clear()


    def clear_recip_arrays(self):
        self.recip_arrs.clear()


    def add_ds_arrays(self, named_arrays, logentry=None):
        names = sorted(list(named_arrays.keys()))
        shape = named_arrays[names[0]].shape
        if not self.are_same_shapes(named_arrays, shape):
            print('arrays in set should have the same shape')
            return
        # find crop beginning and ending
        [(x1, x2), (y1, y2), (z1, z2)] = self.get_crop_points(shape)
        for name in named_arrays.keys():
            self.dir_arrs[name] = named_arrays[name][x1:x2, y1:y2, z1:z2]
        if (not self.dirspace_uptodate):
            self.update_dirspace((x2 - x1, y2 - y1, z2 - z1), shape)


    def are_same_shapes(self, arrays, shape):
        for name in arrays.keys():
            arr_shape = arrays[name].shape
            for i in range(len(shape)):
                if arr_shape[i] != shape[i]:
                    return False
        return True


    def get_crop_points(self, shape):
        # shape and crop should be 3 long
        crop_points = []
        for i in range(len(shape)):
            cropped_size = int(shape[i] * self.crop[i])
            chopped = int((shape[i] - cropped_size) / 2)
            crop_points.append((chopped, chopped + cropped_size))
        return crop_points


    def get_ds_structured_grid(self, **args):
        sg = tvtk.StructuredGrid()
        arr0 = self.dir_arrs[list(self.dir_arrs.keys())[0]]
        dims = list(arr0.shape)
        sg.points = self.dir_coords
        for a in self.dir_arrs.keys():
            arr = tvtk.DoubleArray()
            arr.from_array(self.dir_arrs[a].ravel())
            arr.name = a
            sg.point_data.add_array(arr)

        sg.dimensions = (dims[2], dims[1], dims[0])
        sg.extent = 0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1
        return sg


    def get_rs_structured_grid(self, **args):
        sg = tvtk.StructuredGrid()
        arr0 = self.recip_arrs[list(self.recip_arrs.keys())[0]]
        dims = list(arr0.shape)
        sg.points = self.recip_coords
        for a in self.recip_arrs.keys():
            arr = tvtk.DoubleArray()
            arr.from_array(self.recip_arrs[a].ravel())
            arr.name = a
            sg.point_data.add_array(arr)

        sg.dimensions = (dims[2], dims[1], dims[0])
        sg.extent = 0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1
        return sg


    def write_directspace(self, filename, **args):
        filename = filename.replace(os.sep, '/')
        sgwriter = tvtk.XMLStructuredGridWriter()
        # sgwriter.file_type = 'binary'
        if filename.endswith(".vtk"):
            sgwriter.file_name = filename
        else:
            sgwriter.file_name = filename + '.vts'
        sgwriter.set_input_data(self.get_ds_structured_grid())
        sgwriter.write()
        print('saved file', filename)


    def write_recipspace(self, filename, **args):
        filename = filename.replace(os.sep, '/')
        sgwriter = tvtk.XMLStructuredGridWriter()
        if filename.endswith(".vtk"):
            sgwriter.file_name = filename
        else:
            sgwriter.file_name = filename + '.vts'
        sgwriter.set_input_data(self.get_rs_structured_grid())
        sgwriter.write()
        print('saved file', filename)


def process_dir(instrument, config_map, rampups, crop, unwrap, make_twin, res_dir_scan):
    """
    Loads arrays from files in results directory. If reciprocal array exists, it will save reciprocal info in tif format. It calls the save_CX function with the relevant parameters.
    Parameters
    ----------
    res_dir_conf : tuple
        tuple of two elements:
        res_dir - directory where the results of reconstruction are saved
        conf_dict - dictionary containing configuration parameters
    Returns
    -------
    nothing
    """
    [res_dir, scan] = res_dir_scan
    save_dir = res_dir.replace('_phasing', '_viz')
    # create dir if it does not exist
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

    # get geometry
    instrument.initialize(config_map, scan)
    geometry = instrument.get_geometry(image.shape)
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

    if support is not None:
        image, support = ut.center(image, support)
    if rampups > 1:
        image = ut.remove_ramp(image, ups=rampups)

    crop = crop + [1.0] * (len(image.shape) - len(crop))
    viz = CXDViz(crop, geometry)
    viz.visualize(image, support, coh, save_dir, unwrap)

    if make_twin:
        image = np.conjugate(np.flip(image))
        if support is not None:
            support = np.flip(support)
            image, support = ut.center(image, support)
        if rampups > 1:
            image = ut.remove_ramp(image, ups=rampups)
        viz.visualize(image, support, coh, save_dir, unwrap, True)


def handle_visualization(experiment_dir, rec_id=None):
    """
    If the image_file parameter is defined, the file is processed and vts file saved. Otherwise this function determines root directory with results that should be processed for visualization. Multiple images will be processed concurrently.
    Parameters
    ----------
    conf_dir : str
        directory where the file will be saved
    Returns
    -------
    nothing
    """
    experiment_dir = experiment_dir.replace(os.sep, '/')
    if not os.path.isdir(experiment_dir):
        print("Please provide a valid experiment directory")
        return("Please provide a valid experiment directory")

    print ('starting visualization process')

    main_conf_file = experiment_dir + '/conf/config'
    if not os.path.isfile(main_conf_file):
        print('main configuration file', main_conf_file, 'does not exist')
        return ('main configuration file', main_conf_file, 'does not exist')
    main_conf_map = ut.read_config(main_conf_file)
    if main_conf_map is None:
        print('Cannont parse main configuration file', main_conf_file)
        return ('Cannot parse main configuration file', main_conf_file)

    # convert configuration files if needed
    if 'converter_ver' not in main_conf_map or conv.get_version() is None or conv.get_version() > main_conf_map[
        'converter_ver']:
        conf_maps = conv.convert(experiment_dir + '/conf')
        main_conf_map = conf_maps['config']

    msg = cohere.verify('config', main_conf_map)
    if len(msg) > 0:
        # the error message is printed in verifier
        return msg

    if not os.path.isfile(experiment_dir + '/conf/config_instr'):
        print('configuration file', experiment_dir + '/conf/config_instr', 'does not exist')
        return ('configuration file', experiment_dir + '/conf/config_instr', 'does not exist')

    instr_conf_map = ut.read_config(experiment_dir + '/conf/config_instr')

    msg = cohere.verify('config_instr', instr_conf_map)
    if len(msg) > 0:
        # the error message is printed in verifier
        return msg

    if 'multipeak' in main_conf_map and main_conf_map['multipeak']:
        mp.process_dir(experiment_dir + '/results_phasing')
    else:
        try:
            instr = importlib.import_module('beamlines.' + main_conf_map['beamline'] + '.instrument')
            instrument = instr.Instrument()
        except:
            print('cannot import beamlines.' + main_conf_map['beamline'] + '.instrument module.')
            return ('cannot import beamlines.' + main_conf_map['beamline'] + '.instrument module.')

        config_map = {}
        if ('separate_scans' in main_conf_map and main_conf_map['separate_scans']) or \
                ('separate_scan_ranges' in main_conf_map and main_conf_map['separate_scan_ranges']):
            config_map['diffractometer'] = instr_conf_map['diffractometer']
            config_map['specfile'] = instr_conf_map['specfile']
            separate = True
        else:
            config_map = instr_conf_map
            separate = False

        if os.path.isfile(experiment_dir + '/conf/config_data'):
            data_config_map = ut.read_config(experiment_dir + '/conf/config_data')
            if 'binning' in data_config_map:
                config_map['binning'] = data_config_map['binning']

        # get the visualization config
        disp_config_map = ut.read_config(experiment_dir + '/conf/config_disp')
        if 'rampups' in disp_config_map:
            rampups = disp_config_map['rampups']
        else:
            rampups = 1

        if 'make_twin' in disp_config_map:
            make_twin = disp_config_map['make_twin']
        else:
            make_twin = False

        if 'crop' in disp_config_map:
            crop = disp_config_map['crop']
        else:
            crop = []

        if 'unwrap' in disp_config_map:
            unwrap = disp_config_map['unwrap']
        else:
            unwrap = False

        if 'results_dir' in disp_config_map:
            results_dir = disp_config_map['results_dir'].replace(os.sep, '/')
        elif separate:
            results_dir = experiment_dir
        elif rec_id is not None:
            results_dir = experiment_dir + '/results_phasing_' + rec_id
        else:
            results_dir = experiment_dir + '/results_phasing'
        # find directories with image.npy file in the root of results_dir
        dirs = []
        for (dirpath, dirnames, filenames) in os.walk(results_dir):
            for file in filenames:
                if file.endswith('image.npy'):
                    dirs.append((dirpath).replace(os.sep, '/'))
        if len(dirs) == 0:
            print ('no image.npy files found in the directory tree', results_dir)
            return ('no image.npy files found in the directory tree', results_dir)

        last_scan = int(main_conf_map['scan'].split(',')[-1].split('-')[-1])
        if separate:
            scans = []
            # the scan that will be used to derive geometry is determined from the scan directory
            for dir in dirs:
                subdir = dir.removeprefix(experiment_dir + '/')
                if subdir.startswith('scan'):
                    scan_dir = subdir.split('/')[0]
                    scans.append(int(scan_dir.removeprefix('scan_').split('-')[-1]))
                else:
                    print('directory', subdir, 'does not start with "scan", using configured scan to parse spec')
                    scans.append(last_scan)
            dirs = list(zip(dirs, scans))
        else:
            dirs = [[dir, last_scan] for dir in dirs]

        if len(dirs) == 1:
            process_dir(instrument, config_map, rampups, crop, unwrap, make_twin, dirs[0])
        else:
            func = partial(process_dir, instrument, config_map, rampups, crop, unwrap, make_twin)
            no_proc = min(cpu_count(), len(dirs))
            with Pool(processes = no_proc) as pool:
               pool.map_async(func, dirs)
               pool.close()
               pool.join()
    print ('done with processing display')
    return ''


def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="experiment directory")
    parser.add_argument("--rec_id", help="alternate reconstruction id")
    args = parser.parse_args()
    handle_visualization(args.experiment_dir, args.rec_id)


if __name__ == "__main__":
    main(sys.argv[1:])

# python run_disp.py experiment_dir