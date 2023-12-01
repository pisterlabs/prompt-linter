# -*- coding: utf-8 -*-
"""
Module to define particular circular tangents in a closed polygon in
:math:`\\mathbb{R}^2`.
"""


# %%
class pckCirclesInPolygon:
    '''Creates an instance of an object that defines circular particles tangent
    in a fractal way inside of a closed polygon in :math:`\\mathbb{R}^2`.

    Attributes:
        coordinates ((n, 2) `numpy.ndarray`): Coordinates of vertices of the\
        polygon.
        depth (`int`): Depth fractal for each triangle that compose the\
            triangular mesh. If this number is not given, then,\
            the fractal generation of circles is done up to a circle\
            reachs a radius to lower than the five percent of the\
            incircle radius. Large values of `depth` might produce internal\
            variables that tend to infinte, then a\
            ``ValueError`` is produced with a warning message\
            ``array must not contain infs or NaNs``.

    Note:
        The class ``pckCirclesInPolygon`` requires\
        `NumPy <http://www.numpy.org/>`_,\
        `Matplotlib <https://matplotlib.org/>`_ and\
        `Triangle <http://dzhelil.info/triangle/>`_

    Examples:
        >>> from numpy import array
        >>> from circlespacking import pckCirclesInPolygon
        >>> coords = array([[1, 1], [2, 5], [4.5, 6], [8, 3], [7, 1], [4, 0]])
        >>> pckCircles = pckCirclesInPolygon(coords)
        >>> pckCircles.__dict__.keys()
        dict_keys(['coordinates', 'depth', 'CDT', 'listCircles'])
    '''

    def __init__(self, coordinates, depth=None):
        '''Method for initializing the attributes of the class.'''
        self.coordinates = coordinates
        self.depth = depth
        # initializing methods
        self.trianglesMesh()
        self.generator()

    def trianglesMesh(self):
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
            >>> from basegeometry import Polygon
            >>> from circlespacking import pckCirclesInPolygon
            >>> coordinates = array([[1, 1], [2, 5], [4.5, 6], [6, 4], [8, 3],
                                     [7, 1], [4.5, 1], [4, 0]])
            >>> polygon = Polygon(coordinates)
            >>> boundCoords = polygon.boundCoords
            >>> pckCircles = pckCirclesInPolygon(boundCoords)
            >>> verts = pckCircles.trianglesMesh()

            >>> from numpy import array
            >>> from basegeometry import Polygon
            >>> from circlespacking import pckCirclesInPolygon
            >>> coordinates = array([[2, 2], [2, 6], [8, 6], [8, 2]])
            >>> polygon = Polygon(coordinates)
            >>> boundCoords= polygon.boundCoords
            >>> pckCircles = pckCirclesInPolygon(boundCoords)
            >>> verts =  pckCircles.trianglesMesh()
        '''
        import numpy as np
        from triangle import triangulate

        # polygon area by applying the gauss equation
        area = 0.5*abs(sum(self.coordinates[:-1, 0] * self.coordinates[1:, 1] -
                           self.coordinates[:-1, 1] * self.coordinates[1:, 0]))
        index = np.arange(len(self.coordinates[:-1]))
        indexSegmts = np.column_stack((index, np.hstack((index[1:], [0]))))
        # Max area of the triangles in the Constrained Delaunay triangulation
        maxArea = np.random.uniform(0.25 * area)
        steinerPts = np.random.uniform(5, 50)
        # constrained Delaunay triangulation
        self.CDT = triangulate(tri={'vertices': self.coordinates[:-1],
                               'segments': indexSegmts},
                               opts='pq20a'+str(maxArea)+'S'+str(steinerPts))
        vertsIndex = self.CDT['vertices']
        trianglesIndex = self.CDT['triangles']
        verts = vertsIndex[trianglesIndex]
        return verts

    def generator(self):
        '''Method to generate circular particles in each triangle of the
        triangular mesh.

        Returns:
            listCircles (`list` of Circle objects): `list` that contain all\
                the circles object packed in the polygon.

        Examples:
            >>> from numpy import array
            >>> from circlespacking import pckCirclesInPolygon
            >>> coords = array([[2, 2], [2, 6], [8, 6], [8, 2]])
            >>> pckCircles = pckCirclesInPolygon(coords)
            >>> lstCircles = pckCircles.generator() # list of circles
        '''

        from basegeometry import Triangle

        vertsTriangles = self.trianglesMesh()  # Triangles mesh in polygon
        self.listCircles = list()
        for vert in vertsTriangles:
            self.listCircles += Triangle(vert).packCircles(depth=self.depth,
                                                           want2plot=False)
        return self.listCircles

    def plot(self, plotTriMesh=False):
        '''Method for show a graphic of the circles generated within of the
        polyhon.

        Parameters:
            plotTriMesh (`bool`): Variable to check if it also want to show\
                the graph of the triangles mesh. The default value is ``False``

        Examples:

            .. plot::

                from numpy import array
                from basegeometry import Polygon
                from circlespacking import pckCirclesInPolygon
                coordinates = array([[1, 1], [2, 5], [4.5, 6], [8, 3], [7, 1],
                                     [4, 0]])
                polygon = Polygon(coordinates)
                boundCoords = polygon.boundCoords
                pckCircles = pckCirclesInPolygon(boundCoords, depth=5)
                pckCircles.plot(plotTriMesh=True)


            >>> from numpy import array
            >>> from basegeometry import Polygon
            >>> from circlespacking import pckCirclesInPolygon
            >>> coordinates = array([[1, 1], [2, 5], [4.5, 6], [6, 4], [8, 3],
                                     [7, 1], [4.5, 1], [4, 0]])
            >>> polygon = Polygon(coordinates)
            >>> boundCoords = polygon.boundCoords
            >>> pckCircles = pckCirclesInPolygon(boundCoords)
            >>> pckCircles.plot()

            >>> from slopegeometry import AnthropicSlope
            >>> from circlespacking import pckCirclesInPolygon
            >>> slopeGeometry = AnthropicSlope(12, [1, 1.5], 10, 10)
            >>> boundCoords = slopeGeometry.boundCoords
            >>> pckCircles = pckCirclesInPolygon(boundCoords)
            >>> pckCircles.plot(plotTriMesh=True)

            .. plot::

                from numpy import array
                from slopegeometry import NaturalSlope
                from circlespacking import pckCirclesInPolygon
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
                pckCircles = pckCirclesInPolygon(boundCoords)
                pckCircles.plot(plotTriMesh=True)
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        from triangle import plot as tplot

        # plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.hstack((self.coordinates[:, 0], self.coordinates[0, 0])),
                np.hstack((self.coordinates[:, 1], self.coordinates[0, 1])),
                '-k', lw=1.5, label='Polygon')
        ax.axis('equal')
        ax.set_xlabel('$x$ distance')
        ax.set_ylabel('$y$ distance')
        ax.grid(ls='--', lw=0.5)
        for circle in self.listCircles:
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

    def frecuencyHist(self):
        '''Method to show the histogram of the diameters of the circular
        particles packed in a closed polygon in :math:`\\mathbb{R}^2`.

        Examples:

            .. plot::

                from numpy import array
                from basegeometry import Polygon
                from circlespacking import pckCirclesInPolygon
                coordinates = array([[1, 1], [2, 5], [4.5, 6], [6, 4], [8, 3],
                                     [7, 1], [4.5, 1], [4, 0]])
                polygon = Polygon(coordinates)
                boundCoords = polygon.boundCoords
                pckCircles = pckCirclesInPolygon(boundCoords, 10)
                pckCircles.frecuencyHist()
        '''

        import numpy as np
        import math
        import matplotlib.pyplot as plt

        # Obtaining diameters histogram
        n = len(self.listCircles)  # simple size
        # Number of bins according to Sturges equation
        numBins = math.floor(1 + math.log(n, 2))
        diams = [circle.diameter for circle in self.listCircles]
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

    def loglogDiagram(self):
        '''Method to show the log-log graph of the diameters and quantities
        of circular particles packed in a closed polygon in
        :math:`\\mathbb{R}^2`.

        Examples:

            .. plot::

                from numpy import array
                from basegeometry import Polygon
                from circlespacking import pckCirclesInPolygon
                coordinates = array([[1, 1], [2, 5], [4.5, 6], [6, 4], [8, 3],
                                     [7, 1], [4.5, 1], [4, 0]])
                polygon = Polygon(coordinates)
                boundCoords = polygon.boundCoords
                pckCircles = pckCirclesInPolygon(boundCoords, 10)
                pckCircles.loglogDiagram()
        '''

        import matplotlib.pyplot as plt
        import numpy as np
        import math

        # Obtaining diameters histogram
        n = len(self.listCircles)  # simple size
        # Number of bins according to Sturges equation
        numBins = math.floor(1 + math.log(n, 2))
        diams = [circle.diameter for circle in self.listCircles]
        bins = np.linspace(min(diams), max(diams), numBins)
        hist, binEdges = np.histogram(diams, bins)
        nonZeroIndx = [i for i, k in enumerate(hist) if k != 0]
        histRed = hist[nonZeroIndx]
        histRedRel = [float(k)/n * 100 for k in histRed]
        nonZeroIndx4Bins = [k+1 for k in nonZeroIndx]
        binEdgesRed = binEdges[nonZeroIndx4Bins]
        d = binEdgesRed
        nD = histRedRel
        # plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.loglog(d, nD, 'ko', basex=2)
        ax.grid(ls='--', lw=0.5)
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
