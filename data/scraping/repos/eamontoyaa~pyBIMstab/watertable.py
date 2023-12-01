# -*- coding: utf-8 -*-
'''
Module for defining the classes related to the water table of a slope stability
problem.
'''


# %%
class WaterTable:
    '''Creates an instance of an object that defines the structure of the water
    table of a slope stability problem. ::

        WaterTable(slopeCoords, watertabDepths, smoothFactor=0)

    The water table is defined as tuples of points where the first value is
    the relative distance from the left most point of the slope and the second
    one is the relative depth from the terrain surface.

    Some rules have to be followed: The distance equal to zero must be
    introduced; the distance equal to the horizontal length of the terrain
    surface must be introduced unless the last depth is zero; if the last input
    depth is equal to zero, the water table will continue with the surface
    shape.

    Attributes:
        slopeCoords ((2, n) `numpy.ndarray`): Coordinates of the slope which is
            supossed to be a closed polygon. It can be gotten from the method
            ``defineBoundary`` either from the classes ``AnthropicSlope`` or
            ``NaturalSlope`` (both in the module ``SlopeGeometry``), however it
            works for any polygon.
        watertabDepths ((2, n) `numpy.ndarray`): Relative coordinates to the
            slope surface of the polyline which defines the watertable. First
            row contains the horizontal relative distances from the left most
            point on the terrain surface and the second row contains the depths
            meassured from the terrain surface.
        smoothFactor (`int`): Value to indicate the B-spline interpolation
            order of the smooter function. If is equal to zero, which is the
            default value, the surface will not be smoothed. It is suggested
            that the smooth factor be equal to 2 or 3 because higher values
            tend to lower the water table due to the smoothing.

    Note:
        The class ``WaterTable`` requires `numpy <http://www.numpy.org/>`_,
        `matplotlib <https://matplotlib.org/>`_ and
        `shapely <https://pypi.python.org/pypi/Shapely>`_.

    Examples:
        >>> from numpy import array
        >>> from pybimstab.slope import AnthropicSlope
        >>> from pybimstab.watertable import WaterTable
        >>> slope = AnthropicSlope(slopeHeight=7.5, slopeDip=[1, 1.5],
        >>>                        crownDist=5, toeDist=5)
        >>> watertabDepths = array([[0, 2, 5, 7, 12, 15],
        >>>                        [2.5, 2.5, 3, 1.5, 0.5, 1]])
        >>> watertable = WaterTable(slopeCoords=slope.coords,
        >>>                         watertabDepths=watertabDepths,
        >>>                         smoothFactor=1)
        >>> watertable.__dict__.keys()
        dict_keys(['slopeCoords', 'watertabDepths', 'smoothFactor', 'coords'])
    '''
    def __init__(self, slopeCoords, watertabDepths, smoothFactor=0):
        '''
        WaterTable(slopeCoords, watertabDepths, smoothFactor=0)
        '''
        self.slopeCoords = slopeCoords
        self.watertabDepths = watertabDepths
        self.smoothFactor = smoothFactor
        # Defining the structure of the water table
        self.defineStructre()

    def defineStructre(self):
        '''Method to define the structure of the water table

        If the polyline which defines the water table intersects the terrain
        surface, it will force the water table keeps on the terrain and not
        above it.

        Returns:
            ((2, n) `numpy.ndarray`): Absolute coordinates of the vertices of\
                the polyline which defines the water table. First row contains\
                the abcsises and the second row contains the ordinates.

        Examples:
            >>> from numpy import array
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.watertable import WaterTable
            >>> slope = AnthropicSlope(slopeHeight=7.5, slopeDip=[1, 1.5],
            >>>                        crownDist=5, toeDist=5)
            >>> watertabDepths = array([[0, 2, 5, 7, 12, 15],
            >>>                        [2.5, 2.5, 3, 1.5, 0.5, 1]])
            >>> watertable = WaterTable(slopeCoords=slope.coords,
            >>>                         watertabDepths=watertabDepths,
            >>>                         smoothFactor=1)
            >>> watertable.defineStructre()
            array([[ 0.        ,  0.20408163,  0.40816327,  0.6122449 ,...],
                   [ 6.875     ,  6.875     ,  6.875     ,  6.875     ,...]])
        '''
        import numpy as np
        from shapely.geometry import LineString

        from pybimstab.smoothcurve import SmoothCurve
        from pybimstab.tools import upperLowerSegm

        # Verifications
        xMaxWaterTab = self.watertabDepths[0].max()
        xMaxSlopeSurf = self.slopeCoords[0].max()
        if np.any(self.watertabDepths < 0):
            raise ValueError('Negative values either in distances or depth ' +
                             'are not allowed')
        elif self.watertabDepths[0][0] != 0:
            raise ValueError('The water table left most point is not zero.' +
                             ' It must be included')
        elif xMaxWaterTab - xMaxSlopeSurf > 1e-2:
            raise ValueError('The water table right most point is too long.' +
                             ' Reduce {} units'.format(
                                 round(xMaxWaterTab - xMaxSlopeSurf, 3)))
        elif xMaxSlopeSurf - xMaxWaterTab > 1e-2 and \
                not self.watertabDepths[1, -1] == 0:
            raise ValueError('The water table right most point is too short.' +
                             ' Increase {} units or intersect the surface at'.
                             format(round(xMaxSlopeSurf - xMaxWaterTab, 3)) +
                             ' the last point')

        # Get the ordinates of the water table points projected on terrain surf
        if self.slopeCoords[0, 1] > self.slopeCoords[0, -2]:
            self.slopeCoords = np.fliplr(self.slopeCoords)
        terrainSurfLS = LineString(self.slopeCoords[:, 1:-2].T)
        yMin, yMax = self.slopeCoords[1].min()/2, self.slopeCoords[1].max()*2
        yAtSurface = list()
        for dist, depth in self.watertabDepths.T:
            vertLine = LineString([(dist, yMin), (dist, yMax)])
            intersection = terrainSurfLS.intersection(vertLine)
            yAtSurface.append(intersection.y)
        # Get the temporal coordinates of the water table
        yAtWaterTab = np.array(yAtSurface) - self.watertabDepths[1]
        waterTabCoords = np.array([self.watertabDepths[0], yAtWaterTab])

        # Create intermediate points and smooth the water table.
        waterTabCoords = np.array(
                SmoothCurve(x=waterTabCoords[0], y=waterTabCoords[1],
                            k=self.smoothFactor, n=50).smoothing)
        # Correct segments of the water table above the terrain surface
        correctWatTab = upperLowerSegm(
                terrainSurfLS, LineString(waterTabCoords.T))
        waterTabCoords = np.array(correctWatTab[1]).T

        setattr(self, 'coords', waterTabCoords)
        return waterTabCoords

    def plot(self):
        '''Method for generating a graphic of the water table and the
        slope.

        Returns:
            (`matplotlib.figure.Figure`): object with the matplotlib structure\
                of the plot. You might use it to save the figure for example.

        Examples:
            >>> from numpy import array
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.watertable import WaterTable
            >>> slope = AnthropicSlope(slopeHeight=7.5, slopeDip=[1, 1.5],
            >>>                        crownDist=5, toeDist=5)
            >>> watertabDepths = array([[0, 2, 5, 7, 12, 15],
            >>>                        [2.5, 2.5, 3, 1.5, 0.5, 1]])
            >>> watertable = WaterTable(slopeCoords=slope.coords,
            >>>                         watertabDepths=watertabDepths,
            >>>                         smoothFactor=0)
            >>> fig = watertable.plot()

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/watertable_example1.svg
                :alt: watertable_example1

            .. only:: html

                :download:`example script<../examples/figuresScripts/watertable_example1.py>`.

            >>> watertable = WaterTable(slopeCoords=slope.coords,
            >>>                         watertabDepths=watertabDepths,
            >>>                         smoothFactor=3)
            >>> watertable.plot()

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/watertable_example2.svg
                :alt: watertable_example2

            .. only:: html

                :download:`example script<../examples/figuresScripts/watertable_example2.py>`.

            >>> from numpy import array
            >>> from pybimstab.slope import NaturalSlope
            >>> from pybimstab.watertable import WaterTable
            >>> terrainCoords = array(
            >>>     [[-2.49, 0.1, 1.7, 3.89, 5.9, 8.12, 9.87, 13.29, 20.29,
            >>>       21.43, 22.28, 23.48, 24.65, 25.17],
            >>>      [18.16, 17.88, 17.28, 15.73, 14.31, 13.58, 13, 3.61, 3.61,
            >>>       3.32, 2.71, 2.23, 1.21, 0.25]])
            >>> slope = NaturalSlope(terrainCoords)
            >>> watertabDepths = array([[0, 5, 10, 15, 20, 25, 27.66],
            >>>                         [8, 7, 6, 3, 1, 1, 0.5]])
            >>> watertable = WaterTable(slopeCoords=slope.coords,
            >>>                         watertabDepths=watertabDepths,
            >>>                         smoothFactor=3)
            >>> fig = watertable.plot()

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/watertable_example3.svg
                :alt: watertable_example3

            .. only:: html

                :download:`example script<../examples/figuresScripts/watertable_example3.py>`.

            >>> from numpy import array
            >>> from pybimstab.slope import NaturalSlope
            >>> from pybimstab.watertable import WaterTable
            >>> terrainCoords = array(
            >>>     [[-2.49, 0.1, 1.7, 3.89, 5.9, 8.12, 9.87, 13.29, 20.29,
            >>>       21.43, 22.28, 23.48, 24.65, 25.17],
            >>>      [18.16, 17.88, 17.28, 15.73, 14.31, 13.58, 13, 3.61, 3.61,
            >>>       3.32, 2.71, 2.23, 1.21, 0.25]])
            >>> slope = NaturalSlope(terrainCoords)
            >>> watertabDepths = array([[0, 5, 10, 15],
            >>>                         [8, 7, 3, 0]])
            >>> watertable = WaterTable(slopeCoords=slope.coords,
            >>>                         watertabDepths=watertabDepths,
            >>>                         smoothFactor=3)
            >>> fig = watertable.plot()

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/watertable_example4.svg
                :alt: watertable_example4

            .. only:: html

                :download:`example script<../examples/figuresScripts/watertable_example4.py>`.
        '''
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Plot body
        ax.plot(self.slopeCoords[0], self.slopeCoords[1], '-k')
        ax.plot(self.coords[0], self.coords[1], color='deepskyblue', lw=0.9)
        # Plot settings
        ax.grid(True, ls='--', lw=0.5)
        ax.set_aspect(1)
        fig.tight_layout()
        return fig


# %%
'''
BSD 2 license.

Copyright (c) 2018, Universidad Nacional de Colombia, Exneyder Andres Montoya
    Araque and Ludger O. Suarez-Burgoa.
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
