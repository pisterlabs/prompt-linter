#!/usr/bin/env python3
from pathlib import Path
from h5py import File as HDFFile
import numpy as np
import operator as op
import click
from coherent_point_drift.least_squares import align
from functools import partial
from itertools import repeat, chain

from .main import cli

def rmsd(nuc, structure="0"):
    for chromo, coords in nuc['structures'][structure]['coords'].items():
        mean = np.mean(coords, axis=0, keepdims=True)
        error = np.linalg.norm((coords - mean), axis=-1)
        rmsds = np.sqrt(np.mean(error ** 2, axis=0))
        positions = nuc['structures'][structure]['particles'][chromo]['positions'][:]
        yield chromo, positions, rmsds

@cli.command("rmsd")
@click.argument("nucs", type=Path, nargs=-1, required=True)
@click.option("--structure", default="0", help="Which structure in the file to read")
@click.option("--position", multiple=True, type=(str, int),
              help="Which positions to look at (or all if none provided)")
def output_rmsd(nucs, structure, position):
    from functools import partial

    nucs_rmsds = []
    for nuc in nucs:
        with HDFFile(nuc, "r") as f:
            nuc_rmsds = {}
            for chromo, positions, rmsds in rmsd(f, structure):
                nuc_rmsds.update(zip(zip(repeat(chromo), positions), rmsds))
            nucs_rmsds.append(nuc_rmsds)
    conserved = set.intersection(*map(set, nucs_rmsds))
    if position:
        positions = filter(partial(op.contains, conserved), position)
    else:
        positions = sorted(conserved)
    for chromo, pos in positions:
        print("{}:{} {}".format(chromo, pos, max(rmsd[chromo, pos] for rmsd in nucs_rmsds)))

@cli.command()
@click.argument("nucs", type=Path, nargs=-1, required=True)
@click.option("--structure", default="0", help="Which structure in the file to read")
@click.option("--cols", type=int, default=1)
def plot_rmsd(nucs, structure, cols):
    import matplotlib.pyplot as plt
    from .util import ceil_div
    from collections import defaultdict

    rmsdss = defaultdict(list)
    for nuc in nucs:
        with HDFFile(nuc, "r") as f:
            for chromosome, pos, rmsds in rmsd(f, structure):
                rmsdss[chromosome].append((pos, rmsds))
    fig, axs = plt.subplots(ceil_div(len(rmsdss), cols), cols, sharex=True, sharey=True)
    if cols == 1:
        # Fix matplotlib's return type
        axs = [axs]
    for ax, (chromosome, data) in zip(chain.from_iterable(axs), sorted(rmsdss.items())):
        for poss, rmsds in data:
            ax.plot(poss, rmsds)
        ax.set_ylabel('\n'.join(("RMSD", chromosome)))
    for ax in axs[-1]:
        ax.set_xlabel("Genome Position (bp)")
    plt.show()
