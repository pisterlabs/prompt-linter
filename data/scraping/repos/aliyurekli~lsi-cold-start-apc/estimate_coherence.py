import argparse

from tuning import CoherenceEstimator

CLI = argparse.ArgumentParser()
CLI.add_argument("series", help="Absolute path of the mpd series txt file")
CLI.add_argument("dimensions", type=int, nargs="+", help="Dimensions of coherence")

if __name__ == '__main__':
    args = CLI.parse_args()

    ce = CoherenceEstimator(series=args.series)
    
    for dimension in args.dimensions:
        cval = ce.estimate(dimension)
        print("%d\t%f" % (dimension, cval))
