#!/usr/bin/env python

# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This user script reads raw data, applies correction related to instrument, and saves prepared data.
This script is written for a specific APS beamline. It reads multiple raw data files in each scan directory, applies
darkfield and whitefield correction if applicable, creates 3D stack for each scan, then alignes and combines with
other scans.
"""

__author__ = "Barbara Frosik"
__docformat__ = 'restructuredtext en'
__all__ = ['handle_prep',
           'main']

import argparse
import os
import sys
import importlib
import convertconfig as conv
import cohere_core as cohere
import util.util as ut
from multipeak import MultPeakPreparer
from prep_helper import SepPreparer, SinglePreparer


def prep_data(prep_obj, **kwargs):
    """
    Creates prep_data.tif file in <experiment_dir>/preprocessed_data directory or multiple prep_data.tif in <experiment_dir>/<scan_<scan_no>>/preprocessed_data directories.
    Parameters
    ----------
    none
    Returns
    -------
    nothingcreated mp
    """
    if hasattr(prep_obj, 'multipeak') and prep_obj.multipeak:
        preparer = MultPeakPreparer(prep_obj)
    elif prep_obj.separate_scan_ranges or prep_obj.separate_scans:
        preparer = SepPreparer(prep_obj)
    else:
        preparer = SinglePreparer(prep_obj)

    batches = preparer.get_batches()
    if len(batches) == 0:
        return 'no scans to process'
    preparer.prepare(batches)

    return ''


def handle_prep(experiment_dir, *args, **kwargs):
    """
    Reads the configuration files and accrdingly creates prep_data.tif file in <experiment_dir>/prep directory or multiple
    prep_data.tif in <experiment_dir>/<scan_<scan_no>>/prep directories.
    Parameters
    ----------
    experimnent_dir : str
        directory with experiment files
    Returns
    -------
    experimnent_dir : str
        directory with experiment files
    """
    experiment_dir = experiment_dir.replace(os.sep, '/')
    # check cofiguration
    print ('preaparing data')
    main_conf_file = experiment_dir + '/conf/config'
    main_conf_map = ut.read_config(main_conf_file)
    if main_conf_map is None:
        print ('cannot read configuration file ' + main_conf_file)
        return 'cannot read configuration file ' + main_conf_file
    # convert configuration files if needed
    if 'converter_ver' not in main_conf_map or conv.get_version() is None or conv.get_version() > main_conf_map['converter_ver']:
        conv.convert(experiment_dir + '/conf')
        #re-parse config
        main_conf_map = ut.read_config(main_conf_file)

    er_msg = cohere.verify('config', main_conf_map)
    if len(er_msg) > 0:
        # the error message is printed in verifier
        return er_msg

    if 'beamline' in main_conf_map:
        beamline = main_conf_map['beamline']
        try:
            beam_prep = importlib.import_module('beamlines.' + beamline + '.prep')
        except Exception as e:
            print(e)
            print('cannot import beamlines.' + beamline + '.prep module.')
            return 'cannot import beamlines.' + beamline + '.prep module.'
    else:
        print('Beamline must be configured in configuration file ' + main_conf_file)
        return 'Beamline must be configured in configuration file ' + main_conf_file

    prep_conf_file = experiment_dir + '/conf/config_prep'
    prep_conf_map = ut.read_config(prep_conf_file)
    if prep_conf_map is None:
        return None
    er_msg = cohere.verify('config_prep', prep_conf_map)
    if len(er_msg) > 0:
        # the error message is printed in verifier
        return er_msg

    data_dir = prep_conf_map['data_dir'].replace(os.sep, '/')
    if not os.path.isdir(data_dir):
        print('data directory ' + data_dir + ' is not a valid directory')
        return 'data directory ' + data_dir + ' is not a valid directory'

    instr_config_map = ut.read_config(experiment_dir + '/conf/config_instr')
    # create BeamPrepData object defined for the configured beamline
    conf_map = main_conf_map
    conf_map.update(prep_conf_map)
    conf_map.update(instr_config_map)
    if 'multipeak' in main_conf_map and main_conf_map['multipeak']:
        conf_map.update(ut.read_config(experiment_dir + '/conf/config_mp'))
    prep_obj = beam_prep.BeamPrepData()
    msg = prep_obj.initialize(experiment_dir, conf_map)
    if len(msg) > 0:
        print(msg)
        return msg

    msg = prep_data(prep_obj)
    if len(msg) > 0:
        print(msg)
        return msg

    print('done with preprocessing')
    return ''


def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="directory where the configuration files are located")
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    handle_prep(experiment_dir)


if __name__ == "__main__":
    exit(main(sys.argv[1:]))
