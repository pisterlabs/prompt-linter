# -*- coding: utf-8 -*-
"""
Module to define particular circular tangents in a closed polygon in
:math:`\\mathbb{R}^2`.
"""

import math

import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
from triangle import triangulate
from triangle import plot as tplot

from pc4bims.basegeom import Triangle, Circle


# %%
class CirclePacking:
    '''Creates an instance of an object that defines circular particles tangent
    in a fractal way inside of a closed polygon in :math:`\\mathbb{R}^2`.

    Attributes
    ----------
        coordinates : (n, 2) `numpy.ndarray`
            Coordinates of vertices of the  polygon.
        depth : `int`
            Depth fractal for each triangle that compose the\
            triangular mesh. If this number is not given, then,\
            the fractal generation of circles is done up to a circle\
            reachs a radius to lower than the five percent of the\
            incircle radius. Large values of `depth` might produce internal\
            variables that tend to infinte, then a\
            ``ValueError`` is produced with a warning message\
            ``array must not contain infs or NaNs``.

    Examples
    --------
        >>> from numpy import array
        >>> from pc4bims.circlepacking import CirclePacking as CP
        >>> coords = array([[1, 1], [2, 5], [4.5, 6], [8, 3], [7, 1], [4, 0]])
        >>> circlePacking = CP(coords)
        >>> circlePacking.__dict__.keys()
        dict_keys(['coordinates', 'depth', 'CDT', 'circlesInPoly'])
    '''

    def __init__(self, coordinates, depth=None):
        '''Method for initializing the attributes of the class.'''
        self.coordinates = coordinates
        self.depth = depth
        # initializing methods
        self.trianglesMesh()
        self.genCircles()

    def trianglesMesh(self):
        '''Method to generate a triangles mesh in a polygon by using
        `Constrained Delaunay triangulation\
         <https://en.wikipedia.org/wiki/Constrained_Delaunay_triangulation>`_.

        Returns
        -------
            verts : (n, 3, 2) `numpy.ndarray`
                Vertices of each triangle that compose the triangular mesh.\
                n means the number of triangles; (3, 2) means the index\
                vertices and the coordinates (x, y) respectively.

        Examples
        --------
            >>> from numpy import array
            >>> from pc4bims.basegeom import Polygon
            >>> from pc4bims.circlepacking import CirclePacking as cp
            >>> coordinates = array([[1, 1], [2, 5], [4.5, 6], [6, 4], [8, 3],
                                     [7, 1], [4.5, 1], [4, 0]])
            >>> polygon = Polygon(coordinates)
            >>> boundCoords = polygon.boundCoords
            >>> circlePacking = CP(boundCoords)
            >>> verts = circlePacking.trianglesMesh()

            >>> from numpy import array
            >>> from pc4bims.basegeom import Polygon
            >>> from pc4bims.circlepacking import CirclePacking as cp
            >>> coordinates = array([[2, 2], [2, 6], [8, 6], [8, 2]])
            >>> polygon = Polygon(coordinates)
            >>> boundCoords= polygon.boundCoords
            >>> circlePacking = CP(boundCoords)
            >>> verts =  circlePacking.trianglesMesh()
        '''

        # polygon area by applying the gauss equation
        self.area = 0.5*abs(sum(self.coordinates[:-1, 0] *
                                self.coordinates[1:, 1] -
                                self.coordinates[:-1, 1] *
                                self.coordinates[1:, 0]))
        index = np.arange(len(self.coordinates[:-1]))
        indexSegmts = np.column_stack((index, np.hstack((index[1:], [0]))))
        # Max area of the triangles in the Constrained Delaunay triangulation
        maxArea = np.random.uniform(0.20 * self.area)
        steinerPts = np.random.uniform(10, 50)
        # constrained Delaunay triangulation
        self.CDT = triangulate(tri={'vertices': self.coordinates[:-1],
                               'segments': indexSegmts},
                               opts='pq20a'+str(maxArea)+'S'+str(steinerPts))
        vertsIndex = self.CDT['vertices']
        trianglesIndex = self.CDT['triangles']
        verts = vertsIndex[trianglesIndex]
        return verts

    def genCircles(self):
        '''Method to generate circular particles in the polygon.

        Returns
        -------
            circlesInPoly : `list` of Circle objects
                `list` that contain all the circles object packed in the\
                polygon.

        Examples
        --------
            >>> from numpy import array
            >>> from pc4bims.basegeom import Polygon
            >>> from pc4bims.circlepacking import CirclePacking as CP
            >>> coords = array([[2, 2], [2, 6], [8, 6], [8, 2]])
            >>> polygon = Polygon(coords)
            >>> boundCoords = polygon.boundCoords
            >>> circlePacking = CP(coords)
            >>> lstCircles = circlePacking.genCircles() # list of circles
        '''

        vertsTriangles = self.trianglesMesh()  # Triangles mesh in polygon
        listCircles = list()
        for vert in vertsTriangles:
            listCircles += Triangle(vert).packCircles(depth=self.depth,
                                                      want2plot=False)
        # moving circles
        self.circlesInPoly = list()
        for circle in listCircles:
            # moving circles
            alpha = np.random.uniform(0, 2*np.pi)
            r_new = np.random.uniform(0.5, 0.8) * circle.radius
            D = circle.radius - r_new
            r = D * np.sqrt(np.random.uniform(0, 1))
            c_new = r * np.array([np.cos(alpha), np.sin(alpha)])+circle.center
            self.circlesInPoly.append(Circle(c_new, r_new))
        return self.circlesInPoly

    def arealProportion(self):
        '''Determine the relation between the area of the circles packed and
        the area the polygon.

        Returns:
        -------
        proportion : `float`
            Quotient between the area of the circles and the polygon

        Examples:
        --------
            >>> from numpy import array
            >>> from pc4bims.slope import NaturalSlope
            >>> from pc4bims.circlepacking import CirclePacking as CP
            >>> surfaceCoords = array([[-2.4900, 18.1614],
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
            >>> slopeGeometry = NaturalSlope(surfaceCoords)
            >>> boundCoords = slopeGeometry.boundCoords
            >>> circlePacking = CP(boundCoords, depth=2)
            >>> circlePacking.arealProportion()

            >>> from numpy import array
            >>> from pc4bims.basegeom import Polygon
            >>> from pc4bims.circlepacking import CirclePacking as CP
            >>> coords = array([[2, 2], [2, 6], [8, 6], [8, 2]])
            >>> polygon = Polygon(coords)
            >>> boundCoords = polygon.boundCoords
            >>> circlePacking = CP(boundCoords)
            >>> circlePacking.arealProportion()
        '''
        proportion = sum([c.area for c in self.circlesInPoly]) / self.area
        return proportion

    def plot(self, plotTriMesh=False):
        '''Method for show a graphic of the circles generated within of the
        polygon.

        Parameters
        ----------
            plotTriMesh : `bool`
                Variable to check if it also want to show the graph of the\
                triangles mesh. Default is ``False``

        Examples
        --------

            .. plot::

                from numpy import array
                from pc4bims.basegeom import Polygon
                from pc4bims.circlepacking import CirclePacking as CP
                coordinates = array([[1, 1], [2, 5], [4.5, 6], [8, 3], [7, 1],
                                     [4, 0]])
                polygon = Polygon(coordinates)
                boundCoords = polygon.boundCoords
                circlePacking = CP(boundCoords, depth=3)
                circlePacking.plot(plotTriMesh=True)

                from numpy import array
                from pc4bims.basegeom import Polygon
                from pc4bims.circlepacking import CirclePacking as CP
                coordinates = array([[1, 1], [2, 5], [4.5, 6], [6, 4], [8, 3],
                                     [7, 1], [4.5, 1], [4, 0]])
                polygon = Polygon(coordinates)
                boundCoords = polygon.boundCoords
                circlePacking = CP(boundCoords, 2)
                circlePacking.plot()

                from pc4bims.slope import AnthropicSlope
                from pc4bims.circlepacking import CirclePacking as CP
                slopeGeometry = AnthropicSlope(12, [1, 1.5], 10, 10)
                boundCoords = slopeGeometry.boundCoords
                circlePacking = CP(boundCoords, 5)
                circlePacking.plot(plotTriMesh=True)

                from numpy import array
                from pc4bims.slope import NaturalSlope
                from pc4bims.circlepacking import CirclePacking as CP
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
                circlePacking = CP(boundCoords, 2)
                circlePacking.plot(plotTriMesh=True)
        '''

        # plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.hstack((self.coordinates[:, 0], self.coordinates[0, 0])),
                np.hstack((self.coordinates[:, 1], self.coordinates[0, 1])),
                '-k', lw=1.5, label='Polygon')
        ax.axis('equal')
        ax.set_xlabel('$x$ distance')
        ax.set_ylabel('$y$ distance')
        for circle in self.circlesInPoly:
            ax.add_patch(plt.Circle(circle.center, circle.radius, fill=True,
                                    lw=1, ec='k', fc='k'))
        # plotting triangular mesh
        if plotTriMesh:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.grid(ls='--', lw=0.5)
            tplot.plot(ax, **self.CDT)
            ax.axis('equal')
        return

    def frecuencyHist(self):
        '''Method to show the histogram of the diameters of the circular
        particles packed in a closed polygon in :math:`\\mathbb{R}^2`.

        Examples
        --------

            .. plot::

                from numpy import array
                from pc4bims.basegeom import Polygon
                from pc4bims.circlepacking import CirclePacking as CP
                coordinates = array([[1, 1], [2, 5], [4.5, 6], [6, 4], [8, 3],
                                     [7, 1], [4.5, 1], [4, 0]])
                polygon = Polygon(coordinates)
                boundCoords = polygon.boundCoords
                circlePacking = CP(boundCoords, 10)
                circlePacking.frecuencyHist()
        '''

        # Obtaining diameters histogram
        n = len(self.circlesInPoly)  # simple size
        # Number of bins according to Sturges equation
        numBins = math.floor(1 + math.log2(n))
        diams = [circle.diameter for circle in self.circlesInPoly]
        bins = np.linspace(min(diams), max(diams), numBins)
        # plotting
        plt.style.use('seaborn-white')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(diams, bins)
        ax.grid(ls='--', lw=0.5)
        ax.set_xlabel('Diameters')
        ax.set_ylabel('Frecuency')
        return

    def loglogDiagram(self, num=None):
        '''Method to show the log-log graph of the diameters and quantities
        of circular particles packed in a closed polygon in
        :math:`\\mathbb{R}^2`.

        Examples
        --------

            .. plot::

                from numpy import array
                from pc4bims.basegeom import Polygon
                from pc4bims.circlepacking import CirclePacking as CP
                coordinates = array([[1, 1], [2, 5], [4.5, 6], [6, 4], [8, 3],
                                     [7, 1], [4.5, 1], [4, 0]])
                polygon = Polygon(coordinates)
                boundCoords = polygon.boundCoords
                circlePacking = CP(boundCoords, 10)
                circlePacking.loglogDiagram()
        '''

        # Obtaining diameters histogram
        n = len(self.circlesInPoly)  # simple size
        # Number of bins according to Sturges equation
        numBins = math.floor(1 + math.log2(n))
        diams = [circle.diameter for circle in self.circlesInPoly]
        bins = np.linspace(min(diams), max(diams), numBins)
        hist, binEdges = np.histogram(diams, bins)
        nonZeroIndx = [i for i, k in enumerate(hist) if k != 0]
        histRed = hist[nonZeroIndx]
        histRedRel = [float(k)/n * 100 for k in histRed]
        nonZeroIndx4Bins = [k+1 for k in nonZeroIndx]
        binEdgesRed = binEdges[nonZeroIndx4Bins]
        if n is None:
            d, nD = binEdgesRed, histRedRel
        else:
            d, nD = binEdgesRed[:num], histRedRel[:num]
        # Logarithmic fit
        dLog2, nDLog2 = np.log2(d), np.log2(nD)
        bLog, mLog = polyfit(dLog2, nDLog2, deg=1)
        nDFitLog = mLog * dLog2 + bLog
        nDFit = 2**nDFitLog
        # plotting loglog
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(d, nD, 'ko', ms=3.5)
        ax.plot(d, nDFit, '--r', label='$f_{(x)}=$'+str(round(mLog, 2)) +
                                       '$x\ $'+str(round(bLog, 2)), lw=.8)
        ax.set_xscale('log', basex=2)
        ax.set_yscale('log', basey=2)
        ax.grid(ls='--', lw=0.5)
        ax.set_xlabel('$\log_{2}\ d$')
        ax.set_ylabel('$\log_{2}\ N_d$')
        ax.legend()
        # ax.axis('equal')
        # ax.set_aspect(1)
        ax.set_xlim((0.5*min(d), 1.5*max(d)))
        ax.set_ylim((0.5*min(nD), 1.5*max(nD)))
        return


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
