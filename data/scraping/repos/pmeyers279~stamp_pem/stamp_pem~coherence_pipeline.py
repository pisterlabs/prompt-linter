#!/usr/bin/env python

from stamp_pem import coherence_functions as cf
import optparse
from stamp_pem import coh_io
import os

os.system('echo $LIGO_DATAFIND_SERFER')


def parse_command_line():
    """
    parse_command_line
    """
    parser = optparse.OptionParser()
    parser.add_option(
        "--channel1", help="channel 1", default='L1:GDS-CALIB_STRAIN',
        type=str)
    parser.add_option(
        "--list", help="file containing channel list", default=None,
        type=str)
    parser.add_option(
        "-s", "--start-time", dest='st', help="start time", type=int,
        default=None)
    parser.add_option(
        "-e", "--end-time", dest='et', help="end time", type=int, default=None)
    parser.add_option(
        "--stride", dest='stride', help="stride for ffts",
        type=float, default=1)
    parser.add_option(
        "--delta-f", dest='df', help="frequency range", type=float, default=1)
    parser.add_option(
        "--fhigh", help="max frequency", type=float, default=None)
    parser.add_option(
        "--frames", help="read from frames as opposed to nds", type=int,
        default=False)
    parser.add_option(
        "-d", help="plot directory", type=str, default="./", dest='dir')
    parser.add_option(
        "--segment-duration", dest='sd', help="segment duration for specgram",
        type=float, default=None)
    parser.add_option(
        "--subsystem", dest='subsystem', type=str, help='subsystem', default=None)
    params, args = parser.parse_args()
    return params

params = parse_command_line()

# read channel list
channels = coh_io.read_list(params.list)

# key is the subsystem in this case
if not params.subsystem:
    for key in channels.keys():

        # calculate coherence from list
        cf.coherence_from_list(params.channel1, channels[key], params.stride,
                               params.st, params.et, frames=params.frames,
                               save=True, pad=True, fhigh=int(params.fhigh),
                               subsystem=key)

        # generate filename used in calculating coherence from list
        filename = coh_io.create_coherence_data_filename(
            params.channel1, key, params.st, params.et)

        # plot coherence matrix from file produced
        cf.plot_coherence_matrix_from_file(
            params.channel1, channels, filename, key)
else:
    cf.coherence_from_list(params.channel1, channels[params.subsystem],
                           params.stride,
                           params.st, params.et, frames=params.frames,
                           save=True, pad=True, fhigh=int(params.fhigh),
                           subsystem=params.subsystem)

    # generate filename used in calculating coherence from list
    filename = coh_io.create_coherence_data_filename(
        params.channel1, params.subsystem, params.st, params.et)

    # plot coherence matrix from file produced
    cf.plot_coherence_matrix_from_file(
        params.channel1, channels, filename, params.subsystem)
