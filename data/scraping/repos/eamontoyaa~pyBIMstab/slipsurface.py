# -*- coding: utf-8 -*-
'''
Module for defining the classes related to the slip surface, either a circular
or a composite geometry (*e.g.* a tortuous failure surface)
'''


class CircularSurface:
    '''Creates an instance of an object that defines the structure of an
    polyline which represents the slip surface of a landslide which geometry
    is a circumference-arc. ::

        CircularSurface(slopeCoords, dist1, dist2, radius, concave=True)

    The arc is defined with two points on the terrain surface and the radius.
    That implies there are two possible solutions; to select which one is
    wanted, it is necessary to modify the variable ``concave``.

    It is possible the arc cuts across the terrain surface in some point
    different to its ends, perhaps because of some swedge in the terrrain or
    the radius is too long. In that case, the method ``defineStructre`` changes
    the attribute ``dist2`` such that it is replaced by the horizontal distance
    of the intersection point.

    Attributes:
        slopeCoords ((2, n `numpy.ndarray`): Coordinates of the vertices
            of the polygon within which the slope mass is defined. It
            is obtained with the method ``defineboundary`` either from the
            classes ``AnthropicSlope`` or ``NaturalSlope`` (module ``slope``).
        dist1 (`int` or `float`): First horizontal distance from the leftmost
            point of the terrain surface (including the crown) where the arc
            is intersected with it.
        dist2 (`int` or `float`): Second horizontal distance from the leftmost
            point of the terrain surface (including the crown) where the arc
            is intersected with it.
        radius (`int` or `float`): Length of the circumference-arc radius.
        concave (`bool`): Logical variable to define if it is wanted that
            the circumference-arc will be concave (upwards), otherwise, it will
            be convexe (downwards). Default value is ``True``.

    Note:
        The class ``CircularSurface`` requires
        `numpy <http://www.numpy.org/>`_,
        `matplotlib <https://matplotlib.org/>`_ and
        `shapely <https://pypi.python.org/pypi/Shapely>`_.

    Examples:
        >>> from numpy import array
        >>> from pybimstab.slope import AnthropicSlope
        >>> from pybimstab.slipsurface import CircularSurface
        >>> slope = AnthropicSlope(slopeHeight=7.5, slopeDip=[1, 1.5],
        >>>                        crownDist=5, toeDist=5)
        >>> surface = CircularSurface(slopeCoords=slope.coords,
        >>>                           dist1=2, dist2=10, radius=9)
        >>> surface.__dict__.keys()
        dict_keys(['slopeCoords', 'dist1', 'dist2', 'radius', 'concave',
                   'point1', 'point2', 'center', 'initAngle',
                   'endAngle', 'coords'])
    '''
    def __init__(self, slopeCoords, dist1, dist2, radius, concave=True):
        '''
        CircularSurface(slopeCoords, dist1, dist2, radius, concave=True)
        '''
        self.slopeCoords = slopeCoords
        self.dist1 = min(dist1, dist2)
        self.dist2 = max(dist1, dist2)
        if self.dist2 > max(slopeCoords[0]):
            self.dist2 = max(slopeCoords[0])
        self.radius = radius
        self.concave = concave
        # Definde structure of the arc
        self.defineStructre()

    def defineStructre(self):
        '''Method to define the structure of the circumference-arc which
        represents the slip surface of a landslide.

        If the arc cuts across the terrain surface in some point different to
        its ends, the attribute ``dist2`` is modified to the horizontal
        distance of the intersection point.

        The returned angles have values betwen
        :math: `\\left[\\pi, -\\pi \\right)`, where the angle equal to zero
        coincides with the vector :math: `\\left(1, 0) \\right)`.

        Returns:
            (`dict`): dictionary with the following outputs.

                - **center** (`tuple`): Coordinates of the circumference-arc\
                    center
                - **endAngle** (`float`): Angle in radians of the vector\
                    that points from the center to the first intersection\
                    between the terrain surface and the circumference-arc.
                - **initAngle** (`float`): Angle in radians of the vector\
                    that points from the center to the first intersection\
                    between the terrain surface and the circumference-arc.
                - **point1** (`tuple`): Coordinates of the first point that\
                    intersects the terrain surface
                - **point2** (`tuple`): Coordinates of the second point that\
                    intersects the terrain surface

        Examples:
            >>> from numpy import array
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.slipsurface import CircularSurface
            >>> slope = AnthropicSlope(slopeHeight=7.5, slopeDip=[1, 1.5],
            >>>                        crownDist=5, toeDist=5)
            >>> surface = CircularSurface(slopeCoords=slope.coords,
            >>>                           dist1=2, dist2=10, radius=9)
            >>> surface.defineStructre()
            {'center': (10.881322862689261, 10.831744386868543),
             'endAngle': -1.668878272858519,
             'initAngle': -2.979016942655663,
             'point1': array([2.   , 9.375]),
             'point2': array([10.   ,  1.875])}

            >>> from numpy import array
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.slipsurface import CircularSurface
            >>> slope = AnthropicSlope(slopeHeight=7.5, slopeDip=[1, 1.5],
            >>>                        crownDist=5, toeDist=5)
            >>> surface = CircularSurface(slopeCoords=slope.coords,
            >>>                           dist1=2, dist2=10, radius=1)
            >>> surface.defineStructre()
            ValueError: separation of points > diameter

            >>> from numpy import array
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.slipsurface import CircularSurface
            >>> slope = AnthropicSlope(slopeHeight=7.5, slopeDip=[1, 1.5],
            >>>                        crownDist=5, toeDist=5)
            >>> surface = CircularSurface(slopeCoords=slope.coords,
            >>>                           dist1=2, dist2=10, radius=6)
            >>> surface.defineStructre()
            ValueError: Radius too short. Increase at least 1.516
        '''
        import numpy as np
        from shapely.geometry import LineString

        # Getting the arc coordinates on the terrain surface
        if self.slopeCoords[0, 1] > self.slopeCoords[0, -2]:
            self.slopeCoords = np.fliplr(self.slopeCoords)
        terrainSurfLS = LineString(self.slopeCoords[:, 1:-2].T)
        yMin, yMax = self.slopeCoords[1].min()/2, self.slopeCoords[1].max()*2
        for name, dist in [('point1', self.dist1),
                           ('point2', self.dist2)]:
            vertLine = LineString([(dist, yMin), (dist, yMax)])
            intersection = terrainSurfLS.intersection(vertLine)
            setattr(self, name, np.array([intersection.x, intersection.y]))
        # delta x, delta y between points
        dx, dy = self.point2 - self.point1
        # dist between points
        pointSep = np.linalg.norm(self.point2 - self.point1)
        # Minimum radius
        minRadius = round(0.5 * pointSep**2 / abs(dx), 3)
        # Verification of minimum distance between both points
        if pointSep > 2*self.radius:
            raise ValueError('separation of points > diameter')
        # Verification of minimum radius
        elif minRadius > self.radius:
            raise ValueError('Radius too short. Increase at least {}'.format(
                    round(minRadius - self.radius, 3)))
        # halfway point
        xHalfPoint = (self.point1[0] + self.point2[0])/2
        yHalfPoint = (self.point1[1] + self.point2[1])/2
        # distance along the mirror line
        d = np.sqrt(self.radius**2 - (0.5*pointSep)**2)
        # Verification of the minimum radius
        if self.concave:
            center = (xHalfPoint - d*dy/pointSep, yHalfPoint + d*dx/pointSep)
        else:
            center = (xHalfPoint + d*dy/pointSep, yHalfPoint - d*dx/pointSep)
        setattr(self, 'center', center)

        # Getting the angles from the center to the extreme points of the arc
        vectInters1 = self.point1 - center
        vectInters2 = self.point2 - center
        angles = np.arctan2([vectInters1[1], vectInters2[1]],
                            [vectInters1[0], vectInters2[0]])
        setattr(self, 'initAngle', angles[0])
        setattr(self, 'endAngle', angles[1])

        # Getting the x and y coordinates of 100 points in the arc
        anglAux = angles[:]
        if self.concave:
            anglAux = np.linspace(self.initAngle % (2*np.pi),
                                  self.endAngle % (2*np.pi), 100)
        xC = [self.center[0] + self.radius*np.cos(theta) for theta in anglAux]
        yC = [self.center[1] + self.radius*np.sin(theta) for theta in anglAux]
        setattr(self, 'coords', np.array([xC, yC]))

        # Verify if the arc intersects the terrain in some point diferent to
        # the initial extreme points. If true, cut it in the intersection-point
        lineArc = LineString(self.coords[:, 1:-1].T)
        intersection = terrainSurfLS.intersection(lineArc)
        if intersection.geom_type == 'Point':  # just one intersection
            self.dist2 = intersection.x
            self.defineStructre()
        elif intersection.geom_type == 'MultiPoint':  # more than one
            self.dist2 = intersection[0].x
            self.defineStructre()

        return {'center': center, 'point1': self.point1, 'point2': self.point2,
                'initAngle': angles[0], 'endAngle': angles[1]}

    def plot(self):
        '''Method for generating a graphic of the circumference-arc and the
        slope.

        Returns:
            (`matplotlib.figure.Figure`): object with the matplotlib structure\
                of the plot. You might use it to save the figure for example.

        Examples:
            >>> from numpy import array
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.slipsurface import CircularSurface
            >>> slope = AnthropicSlope(slopeHeight=7.5, slopeDip=[1, 1.5],
            >>>                        crownDist=5, toeDist=5)
            >>> surface = CircularSurface(slopeCoords=slope.coords,
            >>>                           dist1=2, dist2=10, radius=9)
            >>> fig = surface.plot()

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/slipsurface_CircularSurface_example1.svg
                :alt: slipsurface_CircularSurface_example1

            .. only:: html

                :download:`example script<../examples/figuresScripts/slipsurface_CircularSurface_example1.py>`.

            >>> from numpy import array
            >>> from pybimstab.slope import NaturalSlope
            >>> from pybimstab.slipsurface import CircularSurface
            >>> terrainCoords = array(
            >>>     [[-2.49, 0.1 , 1.7 , 3.89, 5.9 , 8.12, 9.87, 13.29, 20.29,
            >>>       21.43, 22.28, 23.48, 24.65, 25.17],
            >>>      [18.16, 17.88, 17.28, 15.73, 14.31, 13.58, 13, 3.61, 3.61,
            >>>       3.32, 2.71, 2.23, 1.21, 0.25]])
            >>> slope = NaturalSlope(terrainCoords)
            >>> surface = CircularSurface(slopeCoords=slope.coords,
            >>>                           dist1=7, dist2=20, radius=13)
            >>> fig = surface.plot()

            .. plot::

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/slipsurface_CircularSurface_example2.svg
                :alt: slipsurface_CircularSurface_example2

            .. only:: html

                :download:`example script<../examples/figuresScripts/slipsurface_CircularSurface_example2.py>`.
        '''
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Plot body
        ax.plot(self.slopeCoords[0], self.slopeCoords[1], '-k')
        ax.plot(self.coords[0], self.coords[1], '-r')
        ax.plot(self.center[0], self.center[1], '.r')
        # Plot settings
        ax.grid(True, ls='--', lw=0.5)
        ax.set_aspect(1)
        fig.tight_layout()
        return fig


# %%
class TortuousSurface:
    '''Creates an instance of an object that defines the structure of an
    polyline which represents the slip surface of a landslide which geometry
    is a tortuous path surrouding the blocks inside the slope mass. ::

        TortuousSurface(bim, dist1, dist2, heuristic='Manhattan',
                        reverseLeft=False, reverseUp=False, smoothFactor=0,
                        preferredPath=None, prefPathFact=None)

    The surface is defined with two points on the terrain surface and the
    heuristic function. It is possible to set a forced path to modify the free
    trajectory of the tortuous path with the aim of move it closer to, for
    example a circular surface.

    Attributes:
        bim (`BlocksInMatrix` object): object with the structure of the slope
            made of the Blocks-In-Matrix material.
        dist1 (`int` or `float`): First horizontal distance from the leftmost
            point of the terrain surface (including the crown) where the arc
            is intersected with it.
        dist2 (`int` or `float`): Second horizontal distance from the leftmost
            point of the terrain surface (including the crown) where the arc
            is intersected with it.
        heuristic (`str`): Name of the geometric model to determine the
            heuristic distance. It must be selected either ``Manhattan`` or
            ``Euclidean``; their description can be found in the ``Astar``
            class documentation. `Manhattan` is the default value.
        reverseLeft (`bool`): Logical variable to allow or not reverses
            movements to the left. Default value is ``False``.
        reverseUp (`bool`): Logical variable to allow or not reverses
            movements to upward. Default value is ``False``.
        smoothFactor (`int`): Value to indicate the B-spline interpolation
            order of the smooter function. If is equal to zero, which is the
            default value, the surface will not be smoothed.
        preferredPath (`numpy.ndarray` or `None`): (2, n) array with the
            coordinates of a path where the tortuous surface is going to be
            forced; ``None`` is the default value.
        prefPathFact (`int` or `float` or `None`): Multiplier of the shortest
            distance between the current point and the polyline; ``None`` is
            the default value.

    Note:
        The class ``TortuousSurface`` requires
        `numpy <http://www.numpy.org/>`_,
        `matplotlib <https://matplotlib.org/>`_ and
        `shapely <https://pypi.python.org/pypi/Shapely>`_.

    Examples:
        >>> from pybimstab.slope import AnthropicSlope
        >>> from pybimstab.bim import BlocksInMatrix
        >>> from pybimstab.slipsurface import TortuousSurface
        >>> slope = AnthropicSlope(slopeHeight=12, slopeDip=[1, 1.5],
        >>>                        crownDist=10, toeDist=10)
        >>> bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.25,
        >>>                      tileSize=0.25, seed=123)
        >>> surface = TortuousSurface(
        >>>     bim, dist1=0, dist2=17, heuristic='manhattan',
        >>>     reverseLeft=False, reverseUp=False, smoothFactor=0,
        >>>     preferredPath=None, prefPathFact=None)
        >>> surface.__dict__.keys()
        dict_keys(['bim', 'dist1', 'dist2', 'heuristic', 'reverseLeft',
                   'reverseUp', 'smoothFactor', 'preferredPath',
                   'prefPathFact', 'terrainSurfLS', 'point1', 'end1', 'point2',
                   'end2', 'startIdx', 'goalIdx', 'coords'])
    '''
    def __init__(self, bim, dist1, dist2, heuristic='manhattan',
                 reverseLeft=False, reverseUp=False, smoothFactor=0,
                 preferredPath=None, prefPathFact=None):
        '''
        TortuousSurface(bim, dist1, dist2, heuristic='Manhattan',
                        reverseLeft=False, reverseUp=False, smoothFactor=0,
                        preferredPath=None, prefPathFact=None)
        '''
        self.bim = bim
        self.dist1 = min(dist1, dist2)
        self.dist2 = max(dist1, dist2)
        if self.dist2 > max(bim.slopeCoords[0]):
            self.dist2 = max(bim.slopeCoords[0])
        self.heuristic = heuristic
        self.reverseLeft = reverseLeft
        self.reverseUp = reverseUp
        self.smoothFactor = smoothFactor
        self.preferredPath = preferredPath
        if preferredPath is not None and prefPathFact is None:
            prefPathFact = 1
        self.prefPathFact = prefPathFact
        # Obtain the indexes at the ends of the slip surface
        self.getIndexesAtEnds()
        # Moving the indexes of the ends when are blocks
        if self.bim.grid[self.startIdx] == 1:
            self.dist1 += self.bim.tileSize
            self.getIndexesAtEnds()
        if self.bim.grid[self.goalIdx] == 1:
            self.dist2 = self.dist2 - self.bim.tileSize
            self.getIndexesAtEnds()
        # Obtain the optimum path through the A* algorithm
        self.defineStructre()

    def getIndexes(self, coord):
        '''Method for obtaining the array indexes of the BIM structure for a
        coordinate given in the real scale of the slope stability problem.

        The transformation is performed by rounding the division between the
        coordinate and the tile size with the ``int_`` function of `numpy`.
        That means that always rounds to the left and bottom sides of a tile.

        Attributes:
            coord (`tuple`): Coordinates of some point in the slope mass or
                surface, which is wanted to get them indexes into the BIM
                grid-grapth structure.

        Returns:
            (`tuple`): Indexes of a coordinate from the real-scale problem\
                projected to the array that represents the BIM structure of\
                the slope; the first value of the tuple is the row and the\
                second one is the column of the grid-array respectively.

        Examples:
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.bim import BlocksInMatrix
            >>> from pybimstab.slipsurface import TortuousSurface
            >>> slope = AnthropicSlope(slopeHeight=12, slopeDip=[1, 1.5],
            >>>                        crownDist=10, toeDist=10)
            >>> bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.25,
            >>>                      tileSize=0.25, seed=123)
            >>> surface = TortuousSurface(bim, dist1=0, dist2=17)
            >>> surface.getIndexes(surface.point1)
            (66, 00)
            >>> surface.getIndexes(surface.point2)
            (24, 68)
        '''
        import numpy as np

        xIdx, yIdx = np.int_(np.array(coord) / self.bim.tileSize)
        return yIdx, xIdx

    def getCoord(self, indexes):
        '''Method for obtaining the real scale problem coordinates of of some
        cell in the BIM grid-graph structure of the slope.

        The transformation is performed by getting the center of the tile which
        contains the coordinates of the point.

        Attributes:
            indexes (`tuple`): Indexes of some cell in the BIM grid-graph
                structure of the slope; the first tuple-value is the ordinate
                and the second one is the abscisse.

        Returns:
            (`tuple`): Indexes of a coordinate from the real-scale problem\
                projected to the array that represents the BIM structure of\
                the slope; the first tuple value is the abscisse and the\
                second one is the ordinate.

        Examples:
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.bim import BlocksInMatrix
            >>> from pybimstab.slipsurface import TortuousSurface
            >>> slope = AnthropicSlope(slopeHeight=12, slopeDip=[1, 1.5],
            >>>                        crownDist=10, toeDist=10)
            >>> bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.25,
            >>>                      tileSize=0.25, seed=123)
            >>> surface = TortuousSurface(bim, dist1=0, dist2=17)
            >>> surface.getCoord((66, 0))
            (0.125, 16.446428571428573)
            >>> surface.getCoord((24, 68))
            (17.125, 5.946428571428573)
        '''
        x = self.bim.xCells[indexes] + 0.5*self.bim.tileSize
        y = self.bim.yCells[indexes] + 0.5*self.bim.tileSize
        return x, y

    def getIndexesAtEnds(self):
        '''Method for obtaining the array indexes of the BIM grid-grapth
        structure for the ends of the slip surface.

        Examples:
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.bim import BlocksInMatrix
            >>> from pybimstab.slipsurface import TortuousSurface
            >>> slope = AnthropicSlope(slopeHeight=12, slopeDip=[1, 1.5],
            >>>                        crownDist=10, toeDist=10)
            >>> bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.25,
            >>>                      tileSize=0.25, seed=123)
            >>> surface = TortuousSurface(bim, dist1=0, dist2=17)
            >>> surface.getIndexesAtEnds()
            ((66, 0), (23, 68))
        '''
        import numpy as np
        from shapely.geometry import LineString

        # Getting the coordinates of the ends
        if self.bim.slopeCoords[0, 1] > self.bim.slopeCoords[0, -2]:
            self.bim.slopeCoords = np.fliplr(self.bim.slopeCoords)
        terrainSurfLS = LineString(self.bim.slopeCoords[:, 1:-2].T)
        setattr(self, 'terrainSurfLS', terrainSurfLS)
        yMin = self.bim.slopeCoords[1].min() / 2
        yMax = self.bim.slopeCoords[1].max() * 2
        tileSize = self.bim.tileSize
        for name, dist in [('point1', self.dist1),
                           ('end1', self.dist1 - (self.dist1 % tileSize)),
                           ('point2', self.dist2),
                           ('end2', self.dist2 - (self.dist2 % tileSize) +
                            tileSize)]:
            vertLine = LineString([(dist, yMin), (dist, yMax)])
            intersection = terrainSurfLS.intersection(vertLine)
            setattr(self, name, np.array([intersection.x, intersection.y]))
        # Indexes of the start and goal nodes
        startIdx = self.getIndexes(self.point1)
        try:  # Controling when index is out the grid dimension
            self.bim.grid[startIdx]
        except Exception:
            startIdx = (startIdx[0]-1, startIdx[1])
        goalIdx = self.getIndexes(self.point2)

        # Moving the indexes of the ends when are out the slope
        while self.bim.grid[startIdx] == -1 or self.bim.grid[goalIdx] == -1:
            # Start index
            if self.bim.grid[startIdx] == -1:
                startIdx = (startIdx[0]-1, startIdx[1])
            if self.bim.grid[goalIdx] == -1:
                goalIdx = (goalIdx[0]-1, goalIdx[1])
        setattr(self, 'startIdx', startIdx)
        setattr(self, 'goalIdx', goalIdx)

        return startIdx, goalIdx

    def defineStructre(self):
        '''Method to define the structure of the tortuous path which represents
        the slip surface of a landslide that occurs in a slope made of BIM.

        The surface is generated through the :math:`\\mathrm{A}^\\ast`
        algorithm defined in the ``Astar`` module.

        Examples:
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.bim import BlocksInMatrix
            >>> from pybimstab.slipsurface import TortuousSurface
            >>> slope = AnthropicSlope(slopeHeight=12, slopeDip=[1, 1.5],
            >>>                        crownDist=10, toeDist=10)
            >>> bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.25,
            >>>                      tileSize=0.25, seed=123)
            >>> surface = TortuousSurface(bim, dist1=0, dist2=17)
            >>> surface.defineStructre()
            array([[ 0.        ,  0.125     ,  0.375     ,  0.625     , ... ],
                   [16.57142857, 16.44642857, 16.19642857, 15.94642857, ... ]])
        '''
        import numpy as np

        from pybimstab.astar import Astar, PreferredPath
        from pybimstab.smoothcurve import SmoothCurve

        # Obtaining the optimum path of the slip surface through A* algorithm
        if self.preferredPath is not None:
            preferredPathIdx = np.array(
                    [self.getIndexes(cr) for cr in self.preferredPath.T]).T
            prefPathAstar = PreferredPath(preferredPathIdx, self.prefPathFact)
        else:
            prefPathAstar = None
        astar = Astar(
            grid=self.bim.grid, startNode=self.startIdx,
            goalNode=self.goalIdx, heuristic=self.heuristic,
            reverseLeft=self.reverseLeft, reverseUp=self.reverseUp,
            preferredPath=prefPathAstar)

        # Transform the optimum path indexes to real-scale coordinates
        coords = [self.getCoord(tuple(idx)) for idx in astar.optimumPath.T]

        # Appendinding the ends to the path
        coords.append(self.end1)
        coords.insert(0, self.end2)

        # Sorting the path such that the begining is in the left side
        coords = np.fliplr(np.array(coords).T)
        if self.smoothFactor > 0:
            smoothedCoords = SmoothCurve(x=coords[0], y=coords[1],
                                         k=self.smoothFactor, n=500)
            coords = smoothedCoords.smoothing
        setattr(self, 'coords', coords)
        return coords

    def plot(self):
        '''Method for generating a graphic of the tortuous slip surface and the
        slope.

        Returns:
            (`matplotlib.figure.Figure`): object with the matplotlib structure\
                of the plot. You might use it to save the figure for example.

        Examples:
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.bim import BlocksInMatrix
            >>> from pybimstab.slipsurface import TortuousSurface
            >>> slope = AnthropicSlope(slopeHeight=12, slopeDip=[1, 1.5],
            >>>                        crownDist=10, toeDist=10)
            >>> bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.25,
            >>>                      tileSize=0.25, seed=123)

            >>> # Not allowing to turn left and up
            >>> surface = TortuousSurface(
            >>>     bim, dist1=0, dist2=17, heuristic='manhattan',
            >>>     reverseLeft=False, reverseUp=False, smoothFactor=0,
            >>>     preferredPath=None, prefPathFact=None)
            >>> fig = surface.plot()

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/slipsurface_TortuousSurface_example1.svg
                :alt: slipsurface_TortuousSurface_example1

            .. only:: html

                :download:`example script<../examples/figuresScripts/slipsurface_TortuousSurface_example1.py>`.

            >>> # Allowing to turn left and up (manhattan heusitic function)
            >>> surface = TortuousSurface(
            >>>     bim, dist1=0, dist2=17, heuristic='manhattan',
            >>>     reverseLeft=True, reverseUp=True, smoothFactor=0,
            >>>     preferredPath=None, prefPathFact=None)
            >>> fig = surface.plot()

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/slipsurface_TortuousSurface_example2.svg
                :alt: slipsurface_TortuousSurface_example2

            .. only:: html

                :download:`example script<../examples/figuresScripts/slipsurface_TortuousSurface_example2.py>`.

            >>> from numpy import array
            >>> from pybimstab.slope import NaturalSlope
            >>> from pybimstab.bim import BlocksInMatrix
            >>> from pybimstab.slipsurface import CircularSurface, TortuousSurface
            >>> terrainCoords = array(
            >>>     [[-2.49, 0.1, 1.7, 3.89, 5.9, 8.12, 9.87, 13.29, 20.29,
            >>>       21.43, 22.28, 23.48, 24.65, 25.17],
            >>>      [18.16, 17.88, 17.28, 15.73, 14.31, 13.58, 13, 3.61, 3.61,
            >>>       3.32, 2.71, 2.23, 1.21, 0.25]])
            >>> slope = NaturalSlope(terrainCoords)
            >>> bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.3,
            >>>                      tileSize=0.35, seed=123)
            >>> preferredPath = CircularSurface(
            >>>     slopeCoords=slope.coords, dist1=5, dist2=15.78, radius=20)

            >>> # With a preferred path and smoothing the surface
            >>> surface = TortuousSurface(
            >>>     bim, dist1=4, dist2=15.78, heuristic='euclidean',
            >>>     reverseLeft=False, reverseUp=False, smoothFactor=2,
            >>>     preferredPath=preferredPath.coords, prefPathFact=2)
            >>> fig = surface.plot()

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/slipsurface_TortuousSurface_example3.svg
                :alt: slipsurface_TortuousSurface_example3

            .. only:: html

                :download:`example script<../examples/figuresScripts/slipsurface_TortuousSurface_example3.py>`.

            >>> # Without a preferred path and smoothing the surface
            >>> surface = TortuousSurface(
            >>>     bim, dist1=5, dist2=15.78, heuristic='euclidean',
            >>>     reverseLeft=False, reverseUp=False, smoothFactor=2,
            >>>     preferredPath=None)
            >>> fig = surface.plot()

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/slipsurface_TortuousSurface_example4.svg
                :alt: slipsurface_TortuousSurface_example4

            .. only:: html

                :download:`example script<../examples/figuresScripts/slipsurface_TortuousSurface_example4.py>`.
        '''
        import numpy as np
        from matplotlib import pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap as newcmap

        # Variables to control the color map and its legend
        if np.any(self.bim.grid == -1):
            cmap = newcmap.from_list('BIMcmap',
                                     ['white', 'lightgray', 'black'], 3)
            ticks = [-1+0.333, 0, 1-0.333]
            ticksLabels = ['None', 'Matrix', 'Blocks']
        else:
            cmap = newcmap.from_list('BIMcmap', ['lightgray', 'black'], 2)
            ticks = [0.25, 0.75]
            ticksLabels = ['Matrix', 'Blocks']
        # Plot body
        fig = plt.figure()
        ax = fig.add_subplot(111)
        bar = ax.pcolormesh(self.bim.xCells, self.bim.yCells, self.bim.grid,
                            cmap=cmap)
        if self.preferredPath is not None:
            ax.plot(self.preferredPath[0], self.preferredPath[1], ':r',
                    lw=2.0, label='Preferred\npath')
        ax.plot(self.coords[0], self.coords[1], '-r', lw=2.5,
                label='Tortuous\nsurface')
        ax.plot(self.bim.slopeCoords[0], self.bim.slopeCoords[1], '-k')
        # Configuring the colorbar
        bar = plt.colorbar(bar, ax=ax, ticks=ticks, pad=0.03,
                           shrink=0.15, aspect=3)
        bar.ax.set_yticklabels(ticksLabels, fontsize='small')
        # Plot settings
        ax.set_aspect(1)
        ax.legend(fontsize='small', bbox_to_anchor=(1.005, 1), loc='best')
        ax.grid(True, ls='--', lw=0.5)
        ax.set_xlim((-0.02*self.bim.slopeCoords[0].max(),
                     1.02*self.bim.slopeCoords[0].max()))
        ax.set_ylim((-0.02*self.bim.slopeCoords[1].max(),
                     1.02*self.bim.slopeCoords[1].max()))
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
