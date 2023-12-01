# -*- coding: utf-8 -*-
"""
Module to define particular circular tangents in a closed polygon in
:math:`\\mathbb{R}^2`.
"""

import math

import numpy as np
import matplotlib.pyplot as plt
from triangle import triangulate
from triangle import plot as tplot

from circpacker.basegeom import Triangle


# %%
class CircPacking:
    '''Creates an instance of an object that defines circular particles tangent
    in a fractal way inside of a closed polygon in :math:`\\mathbb{R}^2`.

    Attributes:
        coordinates ((n, 2) `numpy.ndarray`): Coordinates of vertices of the\
        polygon.
        depth (`int`): Depth fractal for each triangle that compose the\
            triangular mesh. Large values of `depth` might produce internal\
            variables that tend to infinite, then a ``ValueError`` is\
            produced with a warning message ``array must not contain infs or\
            NaNs``.
        minAngle (`int` or `float`): Minimum angle for each triangle of the\
            Delaunay triangulation.
        maxArea (`int` or `float`): Maximum area for each triangle of the\
            Delaunay triangulation.
        length (`int` or `float`): Characteristic length This variable is\
            used to model bimsoils/bimrock. The default value is None.

    Note:
        The class ``CircPacking`` requires\
        `NumPy <http://www.numpy.org/>`_,\
        `Matplotlib <https://matplotlib.org/>`_ and\
        `Triangle <http://dzhelil.info/triangle/>`_

    Examples:
        >>> from numpy import array
        >>> from circpacker.packer import CircPacking as cp
        >>> coords = array([[1, 1], [2, 5], [4.5, 6], [8, 3], [7, 1], [4, 0]])
        >>> pckCircles = cp(coords, depth=5)
        >>> pckCircles.__dict__.keys()
        dict_keys(['coordinates', 'minAngle', 'maxArea', 'lenght', 'depth',
                   'CDT', 'listCirc'])
    '''

    def __init__(self, coordinates, minAngle=None, maxArea=None, length=None,
                 depth=None):
        '''Method for initializing the attributes of the class.'''
        self.coordinates = coordinates
        self.minAngle = minAngle
        self.maxArea = maxArea
        self.lenght = length
        self.depth = depth
        # initializing methods
        self.triMesh()
        self.generator()

    def triMesh(self):
        '''Method to generate a triangles mesh in a polygon by using
        `Constrained Delaunay triangulation\
         <https://en.wikipedia.org/wiki/Constrained_Delaunay_triangulation>`_.

        Return:
            verts ((n, 3, 2) `numpy.ndarray`): Vertices of each triangle that\
                compose the triangular mesh. n means the number of triangles;\
                (3, 2) means the index vertices and the coordinates (x, y)\
                respectively.

        Examples:
            >>> from numpy import array
            >>> from circpacker.basegeom import Polygon
            >>> from circpacker.packer import CircPacking as cp
            >>> coordinates = array([[1, 1], [2, 5], [4.5, 6], [6, 4], [8, 3],
                                     [7, 1], [4.5, 1], [4, 0]])
            >>> polygon = Polygon(coordinates)
            >>> boundCoords = polygon.boundCoords
            >>> circPack = cp(boundCoords, depth=8)
            >>> verts = circPack.triMesh()

            >>> from numpy import array
            >>> from circpacker.basegeom import Polygon
            >>> from circpacker.packer import CircPacking as cp
            >>> coordinates = array([[2, 2], [2, 6], [8, 6], [8, 2]])
            >>> polygon = Polygon(coordinates)
            >>> boundCoords= polygon.boundCoords
            >>> circPack = cp(boundCoords, depth=3)
            >>> verts =  circPack.triMesh()
        '''

        index = np.arange(len(self.coordinates[:-1]))
        indexSegmts = np.column_stack((index, np.hstack((index[1:], [0]))))
        # constrained Delaunay triangulation
        if self.maxArea is None and self.minAngle is None:
            self.CDT = triangulate(tri={'vertices': self.coordinates[:-1],
                                        'segments': indexSegmts},
                                   opts='pq25S15')
        else:
            self.CDT = triangulate(tri={'vertices': self.coordinates[:-1],
                                        'segments': indexSegmts},
                                   opts='pq'+str(self.minAngle)+'a' +
                                   str(self.maxArea))
        vertsIndex = self.CDT['vertices']
        trianglesIndex = self.CDT['triangles']
        verts = vertsIndex[trianglesIndex]
        return verts

    def generator(self):
        '''Method to generate circular particles in each triangle of the
        triangular mesh.

        Returns:
            listCirc (`list` of Circle objects): `list` that contain all\
                the circles object packed in the polygon.

        Examples:
            >>> from numpy import array
            >>> from circpacker.packer import CircPacking as cp
            >>> coords = array([[2, 2], [2, 6], [8, 6], [8, 2]])
            >>> circPack = cp(coords, depth=4)
            >>> lstCircles = circPack.generator() # list of circles
        '''

        vertsTriangles = self.triMesh()  # Triangles mesh in polygon
        self.listCirc = list()
        for v in vertsTriangles:
            self.listCirc += Triangle(v).circInTriangle(depth=self.depth,
                                                        lenght=self.lenght,
                                                        want2plot=False)
        return self.listCirc

    def plot(self, plotTriMesh=False):
        '''Method for show a graphic of the circles generated within of the
        polyhon.

        Parameters:
            plotTriMesh (`bool`): Variable to check if it also want to show\
                the graph of the triangles mesh. The default value is ``False``

        Examples:

            .. plot::

                from numpy import array
                from circpacker.basegeom import Polygon
                from circpacker.packer import CircPacking
                coordinates = array([[1, 1], [2, 5], [4.5, 6], [8, 3], [7, 1],
                                     [4, 0]])
                polygon = Polygon(coordinates)
                boundCoords = polygon.boundCoords
                CircPack = CircPacking(boundCoords, depth=10)
                CircPack.plot(plotTriMesh=True)


            >>> from numpy import array
            >>> from circpacker.basegeom import Polygon
            >>> from circpacker.packer import CircPacking as cp
            >>> coordinates = array([[1, 1], [2, 5], [4.5, 6], [6, 4], [8, 3],
                                     [7, 1], [4.5, 1], [4, 0]])
            >>> polygon = Polygon(coordinates)
            >>> boundCoords = polygon.boundCoords
            >>> pckCircles = cp(boundCoords, depth=8)
            >>> pckCircles.plot()

            >>> from circpacker.slopegeometry import AnthropicSlope
            >>> from circpacker.packer import CircPacking as cp
            >>> slopeGeometry = AnthropicSlope(12, [1, 1.5], 10, 10)
            >>> boundCoords = slopeGeometry.boundCoords
            >>> pckCircles = cp(boundCoords, depth=3)
            >>> pckCircles.plot(plotTriMesh=True)

            .. plot::

                from numpy import array
                from circpacker.slopegeometry import NaturalSlope
                from circpacker.packer import CircPacking as cp
                surfaceCoords = array([[-2.4900, 18.1614],
                                       [0.1022, 17.8824],
                                       [1.6975, 17.2845],
                                       [3.8909, 15.7301],
                                       [5.8963, 14.3090],
                                       [8.1183, 13.5779],
                                       [9.8663, 13.0027],
                                       [13.2865, 3.6058],
                                       [20.2865, 3.6058],
                                       [21.4347, 3.3231],
                                       [22.2823, 2.7114],
                                       [23.4751, 2.2252],
                                       [24.6522, 1.2056],
                                       [25.1701, 0.2488]])
                slopeGeometry = NaturalSlope(surfaceCoords)
                boundCoords = slopeGeometry.boundCoords
                pckCircles = cp(boundCoords, depth=6)
                pckCircles.plot(plotTriMesh=True)
        '''

        # plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.hstack((self.coordinates[:, 0], self.coordinates[0, 0])),
                np.hstack((self.coordinates[:, 1], self.coordinates[0, 1])),
                '-k', lw=1.5, label='Polygon')
        ax.axis('equal')
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')
        ax.grid(ls='--', lw=0.5)
        for circle in self.listCirc:
            ax.add_patch(plt.Circle(circle.center, circle.radius, fill=False,
                                    lw=1, ec='black'))
        # plotting triangular mesh
        if plotTriMesh:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.grid(ls='--', lw=0.5)
            tplot.plot(ax, **self.CDT)
            ax.axis('equal')
        return

    def frecHist(self):
        '''Method to show the histogram of the diameters of the circular
        particles packed in a closed polygon in :math:`\\mathbb{R}^2`.

        Examples:

            .. plot::

                from numpy import array
                from circpacker.basegeom import Polygon
                from circpacker.packer import CircPacking as cp
                coordinates = array([[1, 1], [2, 5], [4.5, 6], [6, 4], [8, 3],
                                     [7, 1], [4.5, 1], [4, 0]])
                polygon = Polygon(coordinates)
                boundCoords = polygon.boundCoords
                circpack = cp(boundCoords, depth=10)
                circpack.frecHist()
        '''

        # Obtaining diameters histogram
        n = len(self.listCirc)  # simple size
        # Number of bins according to Sturges equation
        numBins = math.floor(1 + math.log(n, 2))
        diams = [circle.diameter for circle in self.listCirc]
        bins = np.linspace(min(diams), max(diams), numBins)
        # plotting
        plt.style.use('seaborn-white')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(diams, bins, color='gray')
        ax.grid(ls='--', lw=0.5)
        ax.set_xlabel('Diámetro [$L$]')
        ax.set_ylabel('Frecuencia [$L$]')
        return fig

    def logDiagram(self):
        '''Method to show the log-log graph of the diameters and quantities
        of circular particles packed in a closed polygon in
        :math:`\\mathbb{R}^2`.

        Examples:

            .. plot::

                from numpy import array
                from circpacker.basegeom import Polygon
                from circpacker.packer import CircPacking as cp
                coordinates = array([[1, 1], [2, 5], [4.5, 6], [6, 4], [8, 3],
                                     [7, 1], [4.5, 1], [4, 0]])
                polygon = Polygon(coordinates)
                boundCoords = polygon.boundCoords
                pckCircles = cp(boundCoords, depth=8)
                pckCircles.logDiagram()
        '''

        # Obtaining diameters histogram
        n = len(self.listCirc)  # simple size
        # Number of bins according to Sturges equation
        numBins = math.floor(1 + math.log(n, 2))
        diams = [circle.diameter for circle in self.listCirc]
        bins = np.linspace(min(diams), max(diams), numBins)
        hist, binEdges = np.histogram(diams, bins)
        nonZeroIndx = [i for i, k in enumerate(hist) if k != 0]
        histRed = hist[nonZeroIndx]
        histRedRel = [float(k)/n * 100 for k in histRed]
        nonZeroIndx4Bins = [k+1 for k in nonZeroIndx]
        binEdgesRed = binEdges[nonZeroIndx4Bins]
        d, nD = binEdgesRed, histRedRel
        # plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(d, nD, 'ko', ms=4, mfc='none')
        ax.set_xscale('log', basex=2)
        ax.set_yscale('log', basey=2)
        ax.set_xlabel('$\log_{2}\ d$')
        ax.set_ylabel('$\log_{2}\ N_d$')
        # ax.legend()
        ax.grid(ls='--', lw=0.5)
        ax.set_xlim((0.5*min(d), 1.5*max(d)))
        ax.set_ylim((0.5*min(nD), 1.5*max(nD)))
        return fig


# %%
'''
BSD 2 license.

Copyright (c) 2018, Universidad Nacional de Colombia, Andres Ariza-Triana
and Ludger O. Suarez-Burgoa.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
