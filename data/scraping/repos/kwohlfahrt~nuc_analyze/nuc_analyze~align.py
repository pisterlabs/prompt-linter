#!/usr/bin/env python3
from pathlib import Path
from h5py import File as HDFFile
import numpy as np
import operator as op
from functools import partial
import click
from coherent_point_drift.least_squares import align as least_squares

from .main import cli

@cli.command()
@click.argument("nuc", type=Path, required=True)
@click.option("--target", default="0", help="Which model to align to")
@click.option("--structure", default="0", help="Which structure in the file to read")
@click.option("--mirror/--no-mirror", default=True, help="Also align mirror images")
def align(nuc, target, structure, mirror=True):
    with HDFFile(nuc, "r+") as f:
        coordss = f['structures'][structure]['coords']
        all_coords = np.concatenate(list(coordss.values()), axis=1)
        if target == 'median':
            ref = np.median(all_coords, axis=0)
        else:
            ref = all_coords[int(target)]
        xforms = list(map(partial(least_squares, ref, mirror=mirror), all_coords))
        for chr, coords in coordss.items():
            coords[:] = np.array(list(map(op.matmul, xforms, coords)), dtype=coords.dtype)
