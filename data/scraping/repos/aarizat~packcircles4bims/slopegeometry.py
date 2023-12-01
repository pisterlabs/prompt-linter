# -*- coding: utf-8 -*-
'''
Module for defining the class related to the slope geometry.
'''


# %%
class AnthropicSlope:
    '''Creates an instance of an object that defines the geometrical frame
    of the slope to perform the analysis.

    The geometry of the slope is as follow:
        * It is a right slope, i.e. its face points to the right side.
        * Crown and toe planes are horizontal.
        * The face of the slope is continuous, ie, it has not berms.

    Attributes:
        slopeHeight (`int` or `float`): Height of the slope, ie, vertical\
            length betwen crown and toe planes.
        slopeDip ((2, ) `tuple`, `list` or `numpy.ndarray`): Both horizontal\
            and vertical components of the slope inclination given in that\
            order.
        crownDist (`int` or `float`): Length of the horizontal plane in\
            the crown of the slope.
        toeDist (`int` or `float`): Length of the horizontal plane in the\
            toe of the slope.
        maxDepth (`int` or `float` or `None`): Length of the maximum depth the\
            slope can reach.

    Note:
        The class ``slopegeometry`` requires `NumPy <http://www.numpy.org/>`_\
        and `Matplotlib <https://matplotlib.org/>`_.

    Examples:
        >>> slopeGeometry = AnthropicSlope(12, [1, 1.5], 10, 10)
        >>> slopeGeometry.__dict__.keys()
        dict_keys(['slopeHeight', 'slopeDip', 'crownDist', 'toeDist',
                   'maxDepth', 'boundCoords'])
        '''

    def __init__(self, slopeHeight, slopeDip, crownDist, toeDist,
                 maxDepth=None):
        '''Method for initializing the attributes of the class.'''
        import numpy as np

        self.slopeHeight = slopeHeight
        self.slopeDip = np.array(slopeDip)
        self.crownDist = crownDist
        self.toeDist = toeDist
        # Obtaining the maximum depth of the slope from the toe
        if maxDepth is None:
            self.maxDepth()
        else:
            self.maxDepth = maxDepth
        # Defining the boundary coordinates
        self.defineBoundary()

    def maxDepth(self):
        '''Method to obtain the maximum depth of a slope where a circular
        slope failure analysis can be performed.

        The maximum depth is such that the biggest circle satisfished the\
        following conditions:
            * It is tangent to the bottom.
            * crosses both the extreme points at the crown and toe.
            * It is orthogonal to the crown plane.

        Returns:
            maxDepth (`int` or `float`): Maximum depth of the slope\
                measured vertically from the toe plane.

        Examples:
            >>> slopeGeometry = AnthropicSlope(12, [1, 1.5], 10, 10)
            >>> slopeGeometry.maxDepth()
            4.571428571428573
        '''
        import numpy as np

        # Origin of auxiliar coordinates at the begin of the slope-toe
        # Coordinates of the end of the slope-toe
        extremeToePointVec = np.array([self.toeDist, 0])
        # Horizontal distance of the slope face
        slopeDist = self.slopeHeight * self.slopeDip[0] / self.slopeDip[1]
        # Coordinates of the begin of the slope-crown
        extremeCrownPointVec = (-(slopeDist+self.crownDist), self.slopeHeight)
        # Distance between the two extreme points
        differenceVec = extremeToePointVec - extremeCrownPointVec
        distExtrPts = np.linalg.norm(differenceVec)
        # Radius of the largest circle
        maximumCircleRadius = distExtrPts/2 * distExtrPts/differenceVec[0]
        # Toe depth is the difference between maximum-circle radius and
        # the slope-height
        maxDepth = maximumCircleRadius - self.slopeHeight
        # Setting the attribute to the instanced object.
        setattr(self, 'maxDepth', maxDepth)
        return maxDepth

    def defineBoundary(self):
        '''Method to obtain the coordinates of the boundary vertices of the
        slope and plot it if it is wanted.

        The origin of the coordinates is in the corner of the bottom with the\
        back of the slope. The coordinates define a close polygon, ie, the\
        first pair of coordinates is the same than the last one.

        Returns:
            (`numpy.ndarray`): Coordinates of the boundary vertices of the\
                slope.

        Examples:
            >>> slopeGeometry = AnthropicSlope(12, [1, 1.5], 10, 10)
            >>> slopeGeometry.defineBoundary()
            array([[  0.        ,   0.        ],
                   [ 28.        ,   0.        ],
                   [ 28.        ,   4.57142857],
                   [ 18.        ,   4.57142857],
                   [ 10.        ,  16.57142857],
                   [  0.        ,  16.57142857],
                   [  0.        ,   0.        ]])
        '''
        import numpy as np

        # Slope-face horizontal projection (horizontal distance)
        slopeDist = self.slopeHeight * self.slopeDip[0] / self.slopeDip[1]
        # Creating the contour
        boundCoords = np.array(
                [[0, 0],
                 [self.crownDist + slopeDist + self.toeDist, 0],
                 [self.crownDist + slopeDist + self.toeDist, self.maxDepth],
                 [self.crownDist + slopeDist, self.maxDepth],
                 [self.crownDist, self.maxDepth + self.slopeHeight],
                 [0, (self.maxDepth + self.slopeHeight)],
                 [0, 0]])
        # Setting the attribute to the instanced object.
        setattr(self, 'boundCoords', boundCoords)

        return boundCoords

    def plotSlope(self):
        '''Method for generating a graphic of the slope boundary.

        Examples:
            >>> slopeGeometry = AnthropicSlope(12, [1, 1.5], 10, 10)
            >>> slopeGeometry.plotSlope()

            .. plot::

                from slopegeometry import AnthropicSlope
                slopeGeometry = AnthropicSlope(12, [1, 1.5], 10, 10)
                slopeGeometry.plotSlope()
        '''
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis('equal')
        ax.plot(self.boundCoords[:, 0], self.boundCoords[:, 1], '-k')
        ax.plot(self.boundCoords[:, 0], self.boundCoords[:, 1], '.k')
        ax.grid(True, ls='--', lw=0.5)
        return


# %%
class NaturalSlope:
    '''Creates an instance of an object that defines the geometrical frame
    of the slope to perform the analysis.

    The geometry of the slope is as follow:
        * It is a right slope, i.e. its face points to the right side.
        * The slope is defined with its surface's coordinates.
        * The surface is defined as a polyline such that each segment's slope\
            are always zero or negative.
        * The coordinates' order is such that the highest (and leftmost) point\
            is the first one, and the lowest (and rightmost) is the last one.

    Attributes:
        surfaceCoords (`numpy.ndarray`): Coordinates of the surface's\
            vertices of the slope.

    Note:
        The class ``NaturalSlope`` requires `NumPy <http://www.numpy.org/>`_\
        and `Matplotlib <https://matplotlib.org/>`_.

    Examples:
        >>> from numpy import array
        >>> surfaceCoords = array([[  0.        ,  16.57142857],
                                   [ 10.        ,  16.57142857],
                                   [ 18.        ,   4.57142857],
                                   [ 28.        ,   4.57142857],
                                   [ 28.        ,   0.        ]])
        >>> slopeGeometry = NaturalSlope(surfaceCoords)
        >>> slopeGeometry.__dict__.keys()
        dict_keys(['surfaceCoords', 'slopeHeight', 'maxDepth', 'boundCoords'])
        '''

    def __init__(self, surfaceCoords):
        '''Method for initializing the attributes of the class.'''
        self.surfaceCoords = surfaceCoords
        self.slopeHeight = surfaceCoords[0, 1] - surfaceCoords[-1, 1]
        # Obtaining the maximum depth of the slope from the toe
        self.maxDepth()
        # Defining the boundary coordinates
        self.defineBoundary()

    def maxDepth(self):
        '''Method to obtain the maximum depth of a slope where a circular
        slope failure analysis can be performed.

        The maximum depth is such that the biggest circle satisfished the\
        following conditions:
            * It is tangent to the bottom.
            * crosses both the extreme points at the crown and toe.
            * It is orthogonal to the crown plane.

        Returns:
            maxDepth (`int` or `float`): Maximum depth of the slope\
                measured vertically from the toe plane.

        Examples:
            >>> from numpy import array
            >>> surfaceCoords = array([[  0.        ,  16.57142857],
                                       [ 10.        ,  16.57142857],
                                       [ 18.        ,   4.57142857],
                                       [ 28.        ,   4.57142857]])
            >>> slopeGeometry = NaturalSlope(surfaceCoords)
            >>> slopeGeometry.maxDepth()
            4.571428571428573
        '''
        import numpy as np

        # Distance between the two extreme points
        differenceVec = self.surfaceCoords[-1] - self.surfaceCoords[0]
        distExtrPts = np.linalg.norm(differenceVec)
        # Radius of the largest circle
        maximumCircleRadius = distExtrPts/2 * distExtrPts/differenceVec[0]
        # Toe depth is the difference between maximum-circle radius and
        # the slope-height
        maxDepth = maximumCircleRadius - self.slopeHeight
        # Setting the attribute to the instanced object.
        setattr(self, 'maxDepth', maxDepth)
        return maxDepth

    def defineBoundary(self):
        '''Method to obtain the coordinates of the boundary vertices of the
        slope and plot it if it is wanted.

        The origin of the coordinates is in the corner of the bottom with the\
        back of the slope. The coordinates define a close polygon, ie, the\
        first pair of coordinates is the same than the last one.

        Returns:
            (`numpy.ndarray`): Coordinates of the boundary vertices of the\
                slope.

        Examples:
            >>> from numpy import array
            >>> surfaceCoords = array([[  0.        ,  16.57142857],
                                       [ 10.        ,  16.57142857],
                                       [ 18.        ,   4.57142857],
                                       [ 28.        ,   4.57142857]])
            >>> slopeGeometry = NaturalSlope(surfaceCoords)
            >>> slopeGeometry.defineBoundary()
            array([[  0.        ,   0.        ],
                   [  0.        ,  16.57142857],
                   [ 10.        ,  16.57142857],
                   [ 18.        ,   4.57142857],
                   [ 28.        ,   4.57142857],
                   [ 28.        ,   0.        ],
                   [  0.        ,   0.        ]])

            >>> from numpy import array
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
            >>> slopeGeometry.defineBoundary()
            array([[  0.00000000e+00,   0.00000000e+00],
                   [  0.00000000e+00,   1.96301237e+01],
                   [  2.59220000e+00,   1.93511237e+01],
                   [  4.18750000e+00,   1.87532237e+01],
                   [  6.38090000e+00,   1.71988237e+01],
                   [  8.38630000e+00,   1.57777237e+01],
                   [  1.06083000e+01,   1.50466237e+01],
                   [  1.23563000e+01,   1.44714237e+01],
                   [  1.57765000e+01,   5.07452373e+00],
                   [  2.27765000e+01,   5.07452373e+00],
                   [  2.39247000e+01,   4.79182373e+00],
                   [  2.47723000e+01,   4.18012373e+00],
                   [  2.59651000e+01,   3.69392373e+00],
                   [  2.71422000e+01,   2.67432373e+00],
                   [  2.76601000e+01,   1.71752373e+00],
                   [  2.76601000e+01,   6.66133815e-16],
                   [  0.00000000e+00,   0.00000000e+00]])
        '''
        import numpy as np

        # Obtaining the origin vector to move the surface
        originVec = np.array([self.surfaceCoords[0, 0],
                              self.surfaceCoords[0, 1] -
                              self.slopeHeight - self.maxDepth])

        # Creating the contour
        extraCoords = np.array([
                [self.surfaceCoords[-1, 0],
                 self.surfaceCoords[-1, 1] - self.maxDepth],
                [self.surfaceCoords[0, 0],
                 self.surfaceCoords[0, 1] - self.slopeHeight - self.maxDepth]])
        boundCoords = \
            np.vstack((self.surfaceCoords, extraCoords)) - originVec
        boundCoords = np.vstack(([0, 0], boundCoords))

        # Setting the attribute to the instanced object.
        setattr(self, 'boundCoords', boundCoords)

        return boundCoords

    def plotSlope(self):
        '''Method for generating a graphic of the slope boundary.

        Examples:
            >>> from numpy import array
            >>> surfaceCoords = array([[  0.        ,  16.57142857],
                                       [ 10.        ,  16.57142857],
                                       [ 18.        ,   4.57142857],
                                       [ 28.        ,   4.57142857]])
            >>> slopeGeometry = NaturalSlope(surfaceCoords)
            >>> slopeGeometry.plotSlope()

            .. plot::

                from numpy import array
                from slopegeometry import NaturalSlope
                surfaceCoords = array([[  0.        ,  16.57142857],
                                       [ 10.        ,  16.57142857],
                                       [ 18.        ,   4.57142857],
                                       [ 28.        ,   4.57142857]])
                slopeGeometry = NaturalSlope(surfaceCoords)
                slopeGeometry.plotSlope()

            >>> from numpy import array
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
            >>> slopeGeometry.plotSlope()

            .. plot::

                from numpy import array
                from slopegeometry import NaturalSlope
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
                slopeGeometry.plotSlope()
        '''
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis('equal')
        ax.plot(self.boundCoords[:, 0], self.boundCoords[:, 1], '-k')
        ax.plot(self.boundCoords[:, 0], self.boundCoords[:, 1], '.k')
        ax.grid(True, ls='--', lw=0.5)
        return


# %%
'''
BSD 2 license.

Copyright (c) 2016, Universidad Nacional de Colombia, Exneyder A.
    Montoya-Araque and Ludger O. Suarez-Burgoa.
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
