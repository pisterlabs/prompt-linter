#!/usr/bin/env python3.5

import argparse
import sys

from artiqDrivers.devices.coherentDds.driver import CoherentDds, CoherentDdsSim
from sipyco.pc_rpc import simple_server_loop
from sipyco.common_args import simple_network_args, init_logger_from_args
from oxart.tools import add_common_args


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default=None,
                        help="serial device. See documentation for how to "
                             "specify a USB Serial Number.")
    parser.add_argument("--simulation", action="store_true",
                        help="Put the driver in simulation mode, even if "
                             "--device is used.")
    parser.add_argument("--clockfreq", default=1e9, type=float,
                        help="clock frequency provided to DDS")
    parser.add_argument("--internal-clock", action="store_true")
    parser.add_argument("--disable-coherence", action="append",
                        help="disable coherent switching (=no phase glitches) "
                             "for a given channel")

    simple_network_args(parser, 4000)
    add_common_args(parser)
    return parser


def main():
    args = get_argparser().parse_args()
    init_logger_from_args(args)

    incoherent_channels = [False]*4
    if args.disable_coherence:
        for arg in args.disable_coherence:
            ch = int(arg)
            if ch < 0 or ch > 3:
                raise ValueError("channel must be in 0-3")
            incoherent_channels[ch] = True

    if not args.simulation and args.device is None:
        print("You need to specify either --simulation or -d/--device "
              "argument. Use --help for more information.")
        sys.exit(1)

    if args.simulation:
        dev = CoherentDdsSim()
    else:
        dev = CoherentDds(addr=args.device, clockFreq=args.clockfreq,
                                    internal_clock=args.internal_clock,
                                    incoherent_channels=incoherent_channels)

    simple_server_loop({"coherentDds": dev}, args.bind, args.port)


if __name__ == "__main__":
    main()
