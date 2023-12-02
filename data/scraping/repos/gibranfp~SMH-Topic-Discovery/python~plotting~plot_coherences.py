#!/usr/bin/env python
# -*- coding: utf-8
#
# Gibran Fuentes-Pineda <gibranfp@unam.mx>
# IIMAS, UNAM
# 2017
#
# -------------------------------------------------------------------------
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -------------------------------------------------------------------------
"""
Creates boxplot from the coherences of a list of topic files.
"""
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
from coherence_utils import *

def plot_coherences(paths, xlabel, labels, rotation, show_flag, path_to_save, tight = False):
    """
    Creates a boxplot of coherences
    """
    if len(paths) != len(labels):
        print "Number of labels doesn't match number of files"
        sys.exit(2)

    print "Ploting coherences for the following files:"
    for l,p in zip(labels, paths):
        print "Label:", l, p

    coherences = read_multiple_coherence_files(paths)
    
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(coherences, showmeans=True, meanline=True)

    ax.set_xticklabels(labels, rotation=rotation)
    plt.ylabel("NPMI")
    plt.xlabel(xlabel)

    if tight:
        plt.subplots_adjust(bottom=0.25)
        
    if show_flag:
        plt.show()

    if path_to_save:
        plt.savefig(path_to_save)
    
def main():
    try:
        parser = argparse.ArgumentParser()
        parser = argparse.ArgumentParser(
           description="Creates a boxplot of topic coherences given a set of files with discovered topics")
        parser.add_argument("-x", "--xlabel", type=str, default="",
                            help="Label to add to the x axis")
        parser.add_argument("-l", "--labels", type=str, default="",
                            help="Labels of each box (number of labels must match number of given files)")
        parser.add_argument("-r", "--rotation", type=float, default=float(0.0),
                            help="Rotation of the labels (default 0)")
        parser.add_argument('-c', '--config_file', action='store_true', help="Reads topic files and labels to plot from a file")
        parser.add_argument('-s', '--show_flag', action='store_true',
                            help="show figure")
        parser.add_argument('-t', '--tight', action='store_true',
                            help="Tight")

        parser.add_argument("-p", "--path_to_save", type=str, default=None,
                            help="file where to save the figure")
        parser.add_argument('files', nargs='+')

        args = parser.parse_args()
        if args.config_file:
            filepaths, labels = read_config_file(args.files[0])
        else:
            filepaths = args.files
            labels = args.labels.split()

        plot_coherences(filepaths, args.xlabel, labels, args.rotation, args.show_flag, args.path_to_save, args.tight)
        
    except SystemExit:
        print "for help use --help"
        sys.exit(2)

if __name__ == "__main__":
    main()
