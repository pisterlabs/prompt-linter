#!/usr/bin/env python3
from util import generateDegradation, degrade, loadAll

def generate(args):
    from functools import partial
    from itertools import starmap
    from numpy.random import seed, random, randint
    from numpy import iinfo
    from pickle import dumps
    from sys import stdout
    from coherent_point_drift.align import globalAlignment

    seed(args.seed)
    reference= random((args.N, args.D))
    stdout.buffer.write(dumps(reference))
    seeds = randint(iinfo('int32').max, size=args.repeats)

    degradations = list(map(partial(generateDegradation, args), seeds))
    transformeds = starmap(partial(degrade, reference), degradations)
    if args.method == 'rigid':
        from coherent_point_drift.align import driftRigid as drift
    elif args.method == 'affine':
        from coherent_point_drift.align import driftAffine as drift
    else:
        raise ValueError("Invalid method: {}".format(args.method))
    fits = map(partial(globalAlignment, reference, w=args.w), transformeds)
    for repeat in zip(degradations, fits):
        stdout.buffer.write(dumps(repeat))

def plot(args):
    from pickle import load
    from sys import stdin
    import matplotlib.pyplot as plt
    from itertools import starmap
    from numpy.random import seed, random
    from coherent_point_drift.geometry import rigidXform, RMSD

    seed(4) # For color choice
    reference = load(stdin.buffer)

    rmsds = []
    fig, ax = plt.subplots(1, 1)
    for degradation, fit in loadAll(stdin.buffer):
        color = random(3)
        degraded = degrade(reference, *degradation)
        ax.scatter(degraded[:, 0], degraded[:, 1], marker='o', color=color, alpha=0.2)
        fitted = rigidXform(degraded, *fit)
        ax.scatter(fitted[:, 0], fitted[:, 1], marker='+', color=color)
        rmsds.append(RMSD(reference, fitted))
    ax.scatter(reference[:, 0], reference[:, 1], marker='D', color='black')
    ax.set_xticks([])
    ax.set_yticks([])

    if len(rmsds) > 1:
        fig, ax = plt.subplots(1, 1)
        ax.violinplot(rmsds)
        ax.set_ylabel("RMSD")

    plt.show()

if __name__ == '__main__':
    from math import pi
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Test random data for 2D and 3D alignment")

    subparsers = parser.add_subparsers(dest="cmd")
    subparsers.required = True
    parser_gen = subparsers.add_parser('generate', aliases=['gen'], help="Generate points and fits")
    parser_gen.set_defaults(func=generate)
    parser_gen.add_argument('N', type=int, help='Number of points')
    parser_gen.add_argument('D', type=int, choices=(2, 3), help='Number of dimensions')
    parser_gen.add_argument('repeats', type=int, help='Number of trials to run')

    parser_gen.add_argument('--drop', nargs=2, type=int, default=(0, 0),
                        help='number of points to exclude from the reference set')
    parser_gen.add_argument('--rotate', nargs=2, type=float, default=(-pi, pi),
                        help='The range of rotations to test')
    parser_gen.add_argument('--translate', nargs=2, type=float, default=(-1.0, 1.0),
                        help='The range of translations to test')
    parser_gen.add_argument('--scale', nargs=2, type=float, default=(0.5, 1.5),
                        help='The range of scales to test')
    parser_gen.add_argument('--noise', nargs=2, type=float, default=(0.01, 0.01),
                            help='The amount of noise to add')
    parser_gen.add_argument('--duplicate', nargs=2, type=int, default=(1, 1),
                            help='The range of multiples for each point in the degraded set')
    parser_gen.add_argument('--seed', type=int, default=4,
                            help='The random seed for generating a degradation')
    parser_gen.add_argument('--method', type=str, choices={'rigid', 'affine'}, default='rigid',
                            help="The alignment method to use")
    parser_gen.add_argument('-w', type=float, default=0.1,
                            help="The 'w' parameter to the alignment function")

    parser_plot = subparsers.add_parser('plot', help="Plot the generated points")
    parser_plot.set_defaults(func=plot)

    args = parser.parse_args()
    args.func(args)
