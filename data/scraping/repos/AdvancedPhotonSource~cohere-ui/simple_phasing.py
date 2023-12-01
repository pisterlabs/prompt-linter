import sys
import argparse
import cohere_core as cohere
import os


def reconstruction(datafile):
        datafile = datafile.replace(os.sep, '/')
        cohere.phasing.reconstruction(datafile,
                algorithm_sequence='1*(20*ER+80*HIO)+20*ER',
                shrink_wrap_trigger=[1, 1],
                twin_trigger=[2],
                progress_trigger=[0, 20])


def main(arg):
        parser = argparse.ArgumentParser()
        parser.add_argument("datafile", help="data file name. It should be either tif file or numpy.")
        args = parser.parse_args()
        reconstruction(args.datafile)


if __name__ == "__main__":
    exit(main(sys.argv[1:]))
