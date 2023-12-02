# -*- coding: utf-8 -*-
'''
Module for defining the class related to the slices and their structure.
'''


# %%
class MaterialParameters:
    '''Creates an instance of an object that defines the structure of the
    material where the stabilibity analysis is performed. ::

        MaterialParameters(cohesion, frictAngle, unitWeight,
                           blocksUnitWeight=None, wtUnitWeight=None)

    Contains the strength parameters (Mohr-Coulomb failure criterion) and the
    different unit weigths.

    Attributes:
        cohesion (`int` or `float`): Intercept of the Mohr-Coulomb envelope.
        frictAngle (`int` or `float`): Angle of the Mohr-Coulomb envelope.
        unitWeight (`int` or `float`): Unit weight of the soil or the matrix in
            the case of a Blocks-in-Matrix material.
        blocksUnitWeight (`int` or `float`): Unit weight of the blocks in the
            case of a Blocks-in-Matrix material. ``None`` is the default value
            that means there are no blocks.
        wtUnitWeight (`int` or `float`): Unit weight of the water. ``None`` is
            the default value that means there are no water table, or it is
            located under the base of a specific slice.

    Examples:
        >>> from pybimstab.slices import MaterialParameters
        >>> material = MaterialParameters(
        >>>     cohesion=15, frictAngle=23, unitWeight=17, blocksUnitWeight=21,
        >>>     wtUnitWeight=9.8)
        >>> material.__dict__
        {'blocksUnitWeight': 21,
         'cohesion': 15,
         'frictAngle': 23,
         'mtxUnitWeight': 17,
         'wtUnitWeight': 9.8}
        '''

    def __init__(self, cohesion, frictAngle, unitWeight, blocksUnitWeight=None,
                 wtUnitWeight=None):
        '''
        MaterialParameters(cohesion, frictAngle, unitWeight,
                           blocksUnitWeight=None, wtUnitWeight=None)
        '''
        self.cohesion = cohesion
        self.frictAngle = frictAngle
        self.unitWeight = unitWeight
        self.blocksUnitWeight = blocksUnitWeight
        self.wtUnitWeight = wtUnitWeight
        if blocksUnitWeight is None:
            self.blocksUnitWeight = 0
        if wtUnitWeight is None:
            self.wtUnitWeight = 0


# %%
class SliceStr:
    '''Creates an instance of an object that defines the structure of a slice
    of the soil mass above the slip surface. ::

        SliceStr(material, terrainLS, slipSurfLS, watertabLS=None, bim=None)

    The structure contains the required data for performing the slope stability
    assessment by the limit equilibrium method.

    Attributes:
        material (`MaterialParameters` object): object with the parameters of
            the material that composes the slice.
        terrainLS (`shapely.geometry.linestring.LineString`): Top of the slice
            that coincides with the terrain surface.
        slipSurfLS (`shapely.geometry.linestring.LineString`): Bottom of the
            slice that coincides with the slip surface.
        watertabLS (`shapely.geometry.linestring.LineString`): Polyline
            that coincides with the water table. It could be located below or
            crossing the base of the slice, between the bottom and the top or
            absent. ``None`` is the default value.
        bim (`BimStructure` object): object with the structure of the slope
            made of the Blocks-in-Matrix material. ``None`` is the default
            value and means that there is not a BIM structure

    Note:
        The class ``SliceStr`` requires
        `copy <https://docs.python.org/3/library/copy.html>`_,
        `numpy <http://www.numpy.org/>`_,
        `matplotlib <https://matplotlib.org/>`_ and
        `shapely <https://pypi.python.org/pypi/Shapely>`_.

    Examples:
        >>> from shapely.geometry import LineString
        >>> from pybimstab.slices import MaterialParameters, SliceStr
        >>> material = MaterialParameters(cohesion=15, frictAngle=23,
        >>>                               unitWeight=17)
        >>> terrainLS = LineString([(6, 8), (7, 6.5)])
        >>> slipSurfLS = LineString([(6, 3.395), (7, 2.837)])
        >>> watertabLS = LineString([(6, 5), (7, 4)])
        >>> slice_ = SliceStr(material, terrainLS, slipSurfLS, watertabLS,
        >>>                   bim=None)
        >>> slice_.__dict__.keys()
        dict_keys(['material', 'terrainLS', 'slipSurfLS', 'watertabLS', 'bim',
                   'terrainCoords', 'slipSurfCoords', 'watertabCoords',
                   'sliceLS', 'coords', 'xMin', 'xMax', 'yMin', 'yMax', 'area',
                   'width', 'baseSlope', 'alpha', 'baseLength', 'l',
                   'topLength', 'topSlope', 'topInclinatDeg', 'midHeight',
                   'midWatTabHeight'])
        '''

    def __init__(self, material, terrainLS, slipSurfLS, watertabLS=None,
                 bim=None):
        '''
        SliceStr(material, terrainLS, slipSurfLS, watertabLS=None, bim=None)
        '''

        self.material = material
        self.terrainLS = terrainLS
        self.slipSurfLS = slipSurfLS
        self.watertabLS = watertabLS
        self.bim = bim
        # Defining the slice boundary and others attributes of the structure
        self.defineStructure()
        # Extracting the BIM structure
        if bim is not None:
            self.extractBim()

    def defineStructure(self):
        '''Method for defining the geometric structure of the slice including
        the variables required to perform the slope stability assessment by the
        limit equilibrium method.

        Returns:
            ((2, n) `numpy.ndarray`): Coordinates of the slice boundary. First\
                row contains the abscises and the second one contains the\
                ordinates.

        Examples:
            >>> from shapely.geometry import LineString
            >>> from pybimstab.slices import MaterialParameters, SliceStr
            >>> material = MaterialParameters(cohesion=15, frictAngle=23,
            >>>                               unitWeight=17)
            >>> terrainLS = LineString([(6, 8), (7, 6.5)])
            >>> slipSurfLS = LineString([(6, 3.395), (7, 2.837)])
            >>> watertabLS = LineString([(6, 5), (7, 4)])
            >>> slice_ = SliceStr(material, terrainLS, slipSurfLS, watertabLS,
            >>>                   bim=None)
            >>> slice_.defineStructure()
            array([[6.   , 7.   , 7.   , 6.   , 6.   ],
                   [8.   , 6.5  , 2.837, 3.395, 8.   ]])
        '''
        import numpy as np
        from numpy.polynomial.polynomial import polyfit
        from shapely.geometry import Polygon

        # Coordinates
        setattr(self, 'terrainCoords', np.array(self.terrainLS.coords).T)
        setattr(self, 'slipSurfCoords', np.array(self.slipSurfLS.coords).T)
        if self.watertabLS is not None:
            setattr(self, 'watertabCoords', np.array(self.watertabLS.coords).T)
        else:
            setattr(self, 'watertabCoords', None)
        sliceLS = Polygon(
            np.hstack((self.terrainCoords, np.fliplr(self.slipSurfCoords))).T)
        setattr(self, 'sliceLS', sliceLS)
        setattr(self, 'coords', np.array(sliceLS.exterior.coords).T)
        setattr(self, 'xMin', sliceLS.bounds[0])
        setattr(self, 'xMax', sliceLS.bounds[2])
        setattr(self, 'yMin', sliceLS.bounds[1])
        setattr(self, 'yMax', sliceLS.bounds[3])

        # Area
        setattr(self, 'area', sliceLS.area)

        # Width
        setattr(self, 'width', self.xMax - self.xMin)

        # Mean inclination of the base
        # (alpha is the base inclination angle with the opposite sign)
        intercept, slope = polyfit(
                self.slipSurfCoords[0], self.slipSurfCoords[1], deg=1)
        setattr(self, 'baseSlope', slope)
        setattr(self, 'alpha', -1 * np.degrees(np.arctan(slope)))

        # Lengths (l is the base length approximate)
        setattr(self, 'baseLength', self.slipSurfLS.length)  # bottom
        setattr(self, 'l', self.width / np.cos(np.radians(self.alpha)))  # l
        setattr(self, 'topLength', self.terrainLS.length)  # top

        # Mean inclination of the top
        intercept, slope = polyfit(
                self.terrainCoords[0], self.terrainCoords[1], deg=1)
        setattr(self, 'topSlope', slope)
        setattr(self, 'topInclinatDeg', np.degrees(np.arctan(slope)))

        # Heights
        leftHeight = self.terrainCoords[1, 0] - self.slipSurfCoords[1, 0]
        rightHeight = self.terrainCoords[1, -1] - self.slipSurfCoords[1, -1]
        midHeight = 0.5 * (leftHeight + rightHeight)
        setattr(self, 'midHeight', midHeight)
        if self.watertabLS is not None:
            leftWatTabHeight = self.watertabCoords[1, 0] - \
                self.slipSurfCoords[1, 0]
            rightWatTabHeight = self.watertabCoords[1, -1] - \
                self.slipSurfCoords[1, -1]
            midWatTabHeight = 0.5 * (leftWatTabHeight + rightWatTabHeight)
            if midWatTabHeight < 0:
                midWatTabHeight = 0
            setattr(self, 'midWatTabHeight', midWatTabHeight)
        else:
            setattr(self, 'midWatTabHeight', 0)
        return self.coords

    def extractBim(self):
        '''Returns an object from the class ``bim.BlocksInMatrix`` where is
        stored the structure of the local Blocks-In-Matrix material inside the
        slice.

        Examples:
            >>> from shapely.geometry import LineString
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.bim import BlocksInMatrix
            >>> from pybimstab.slices import MaterialParameters, SliceStr
            >>> material = MaterialParameters(cohesion=15, frictAngle=23,
            >>>                               unitWeight=17)
            >>> slope = AnthropicSlope(slopeHeight=7.5, slopeDip=[1, 1.5],
            >>>                        crownDist=5, toeDist=5, depth=2)
            >>> bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.2,
            >>>                      tileSize=0.5, seed=123)
            >>> terrainLS = LineString([(6, 8), (7, 6.5)])
            >>> slipSurfLS = LineString([(6, 3.395), (7, 2.837)])
            >>> watertabLS = LineString([(6, 5), (7, 4)])
            >>> slice_ = SliceStr(material, terrainLS, slipSurfLS, watertabLS,
            >>>                   bim=bim)
            >>> localBIM = slice_.extractBim()
            >>> localBIM.__dict__.keys()
            dict_keys(['slopeCoords', 'blockProp', 'tileSize', 'seed', 'grid',
                       'xCells', 'yCells'])
        '''
        from copy import deepcopy

        import numpy as np

        localBIM = deepcopy(self.bim)  # copy of the original object of the BIM

        # Getting the extreme indexes for extracting the arrays
        xMinIdx, yMinIdx, xMaxIdx, yMaxIdx = np.int_(
                np.array(self.sliceLS.bounds) / localBIM.tileSize)
        # Extracting the BIM structure
        localBIM.grid = localBIM.grid[yMinIdx:yMaxIdx, xMinIdx:xMaxIdx]
        localBIM.xCells = localBIM.xCells[yMinIdx:yMaxIdx+1, xMinIdx:xMaxIdx+1]
        localBIM.yCells = localBIM.yCells[yMinIdx:yMaxIdx+1, xMinIdx:xMaxIdx+1]
        setattr(self, 'localBIM', localBIM)
        # Counting the number of blocks
        numBlocks = np.sum(self.localBIM.grid == 1)
        setattr(self, 'numBlocks', numBlocks)
        return localBIM

    def plot(self):
        '''Method for plotting the slice.

        Examples:
            >>> from shapely.geometry import LineString
            >>> from pybimstab.slices import MaterialParameters, SliceStr
            >>> material = MaterialParameters(cohesion=15, frictAngle=23,
            >>>                               unitWeight=17)
            >>> terrainLS = LineString([(6, 8), (7, 6.5)])
            >>> slipSurfLS = LineString([(6, 3.395), (7, 2.837)])
            >>> watertabLS = LineString([(6, 5), (7, 4)])
            >>> slice_ = SliceStr(material, terrainLS, slipSurfLS, watertabLS,
            >>>                   bim=None)
            >>> fig = slice_.plot()

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/slices_SliceStr_example1.svg
                :alt: slices_SliceStr_example1

            .. only:: html

                :download:`example script<../examples/figuresScripts/slices_SliceStr_example1.py>`.

            >>> from shapely.geometry import LineString
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.bim import BlocksInMatrix
            >>> from pybimstab.slices import MaterialParameters, SliceStr
            >>> material = MaterialParameters(cohesion=15, frictAngle=23,
            >>>                               unitWeight=17)
            >>> slope = AnthropicSlope(slopeHeight=7.5, slopeDip=[1, 1.5],
            >>>                        crownDist=5, toeDist=5, depth=2)
            >>> bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.2,
            >>>                      tileSize=0.5, seed=123)
            >>> terrainLS = LineString([(6, 8), (7, 6.5)])
            >>> slipSurfLS = LineString([(6, 3.395), (7, 2.837)])
            >>> watertabLS = LineString([(6, 5), (7, 4)])
            >>> slice_ = SliceStr(material, terrainLS, slipSurfLS, watertabLS,
            >>>                   bim=bim)
            >>> fig = slice_.plot()

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/slices_SliceStr_example2.svg
                :alt: slices_SliceStr_example2

            .. only:: html

                :download:`example script<../examples/figuresScripts/slices_SliceStr_example2.py>`.
        '''
        import numpy as np
        from matplotlib import pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap as newcmap

        # Variables to control the color map and its legend
        if self.bim is not None:
            if np.any(self.localBIM.grid == -1):
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
        if self.bim is not None:
            bar = ax.pcolormesh(self.localBIM.xCells, self.localBIM.yCells,
                                self.localBIM.grid, cmap=cmap)
            # Configuring the colorbar
            bar = plt.colorbar(bar, ax=ax, ticks=ticks, pad=0.025,
                               shrink=0.15, aspect=3)
            bar.ax.set_yticklabels(ticksLabels, fontsize='small')
        if self.watertabCoords is not None:
            ax.plot(self.watertabCoords[0], self.watertabCoords[1],
                    'deepskyblue', label='watertable')
        ax.plot(self.coords[0], self.coords[1], '-r', label='slice')
        # Plot settings
        ax.set_aspect(1)
        ax.legend(fontsize='small', bbox_to_anchor=(1.005, 1), loc=2)
        ax.grid(True, ls='--', lw=0.5)
        ax.set_xlim((self.coords[0].min() - 0.02*self.coords[0].max(),
                     1.02*self.coords[0].max()))
        ax.set_ylim((self.coords[1].min() - 0.02*self.coords[1].max(),
                     1.02*self.coords[1].max()))
        fig.tight_layout()
        return fig


# %%
class Slices:
    '''Creates an instance of an object that defines the structure of the
    slices that the soil mass above the slip surface is divided. ::

        Slices(material, slipSurfCoords, slopeCoords, numSlices=20,
               watertabCoords=None, bim=None)

    Attributes:
        slipSurfCoords ((2, n) `numpy.ndarray`): Absolute coordinates of the
            vertices of the polyline which defines the slip surface. First row
            contains the abcsises and the second row contains the ordinates.
        slopeCoords ((n, 2) `numpy.ndarray`): Absolute coordinates of the
            vertices of the polyline which defines the slip surface. First
            column contains the abcsises and the second one contains the
            ordinates.
        material (`MaterialParameters` object): object with the
            parameters of the material that composes the slope.
        numSlices (`int`): Number of slices in which the soil above the slip
            surface is going to be divided. 20 is the default value.
        watertabCoords ((2, n) `numpy.ndarray`): Absolute coordinates of the
            vertices of the polyline which defines the water table. First row
            contains the abcsises and the second row contains the ordinates.
            It is like an optional argument and if there is not a water table,
            set it as ``None``, which is the default value.
        bim (`BimStructure` object): object with the structure of
            the slope made of the BIM-material. It is like an optional
            argument and if there is not a BIM structure, set it as ``None``,
            which is the default value.

    Note:
        The class ``Slices`` requires `numpy <http://www.numpy.org/>`_,
        `scipy <https://www.scipy.org/>`_,
        `matplotlib <https://matplotlib.org/>`_ and
        `shapely <https://pypi.python.org/pypi/Shapely>`_.

    Examples:
        >>> from numpy import array
        >>> from pybimstab.slope import AnthropicSlope
        >>> from pybimstab.slipsurface import CircularSurface
        >>> from pybimstab.slices import MaterialParameters, Slices
        >>> slope = AnthropicSlope(slopeHeight=7.5, slopeDip=[1, 1.5],
        >>>                        crownDist=5, toeDist=5)
        >>> surface = CircularSurface(slopeCoords=slope.coords,
        >>>                           dist1=2, dist2=10, radius=9)
        >>> material = MaterialParameters(
        >>>     cohesion=15, frictAngle=23, unitWeight=17,
        >>>     blocksUnitWeight=21, wtUnitWeight=9.8)
        >>> slices = Slices(
        >>>     material=material, slipSurfCoords=surface.coords,
        >>>     slopeCoords=slope.coords, numSlices=10,
        >>>     watertabCoords=None, bim=None)
        >>> slices.__dict__.keys()
        dict_keys(['material', 'numSlices', 'slipSurfCoords', 'slopeCoords',
                   'watertabCoords', 'bim', 'rotationPt', 'slices'])
        '''

    def __init__(self, material, slipSurfCoords, slopeCoords, numSlices=20,
                 watertabCoords=None, bim=None):
        '''
        Slices(material, slipSurfCoords, slopeCoords, numSlices=20,
               watertabCoords=None, bim=None)
        '''
        self.material = material
        self.numSlices = numSlices
        self.slipSurfCoords = slipSurfCoords
        self.slopeCoords = slopeCoords
        self.watertabCoords = watertabCoords
        self.bim = bim
        # Defining the structure of the water table
        self.fitCircle()
        self.createSlices()
        self.setExtLoads()

    def fitCircle(self):
        '''Method for adjusting a circumference to a cloud of points.

        Returns:
            (`dict`): Dictionary with the radius and coordinates of center.

        Examples:
            >>> from numpy import array
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.slipsurface import CircularSurface
            >>> from pybimstab.slices import MaterialParameters, Slices
            >>> slope = AnthropicSlope(slopeHeight=7.5, slopeDip=[1, 1.5],
            >>>                        crownDist=5, toeDist=5)
            >>> surface = CircularSurface(slopeCoords=slope.coords,
            >>>                           dist1=2, dist2=10, radius=9)
            >>> material = MaterialParameters(
            >>>     cohesion=15, frictAngle=23, unitWeight=17,
            >>>     blocksUnitWeight=21, wtUnitWeight=9.8)
            >>> slices = Slices(
            >>>     material=material, slipSurfCoords=surface.coords,
            >>>     slopeCoords=slope.coords, numSlices=10,
            >>>     watertabCoords=None, bim=None)
            >>> slices.fitCircle()
            {'center': array([10.88132286, 10.83174439]),
             'radius': 9.000000000000002,
             'dist1': 2.009892716098348,
             'dist2': 11.761811095385543}
        '''
        import numpy as np
        from scipy import optimize
        from shapely.geometry import LineString, Point

        x, y = self.slipSurfCoords

        def radius(xc, yc):
            '''Gets the distance of each points from the center (xc, yc)'''
            return np.sqrt((x-xc)**2 + (y-yc)**2)

        def f_2(c):
            '''Calculates the algebraic distance between the data points and
            the mean circle centered at c=(xc, yc)'''
            Ri = radius(*c)
            return Ri - Ri.mean()

        estimatedCenter = 3 * np.mean(self.slipSurfCoords, 1)
        center, __ = optimize.leastsq(f_2, estimatedCenter)
        setattr(self, 'rotationPt', center)
        allRadius = radius(*center)
        meanRadius = allRadius.mean()  # radius of fitted circle
        # Create the circle:
        centerPt = Point(*center)
        circleLs = centerPt.buffer(meanRadius).boundary
        # Creating the terrain surface and the intersection with circleLs
        terrainSurfLS = LineString(self.slopeCoords[:, 1:-2].T)
        intersections = circleLs.intersection(terrainSurfLS)
        if intersections.geom_type == 'MultiPoint':
            dist1, dist2 = intersections[0].x, intersections[-1].x
        else:
            dist1, dist2 = x.min(), x.max()
        fittedCirc = {'center': center, 'radius': meanRadius,
                      'dist1': dist1, 'dist2': dist2}
        setattr(self, 'fittedCirc', fittedCirc)
        return fittedCirc

    def createSlices(self):
        '''Method for defining the structure of all the slices in which the
        soil mass above the slip surface is divided.

        Returns:
            (`list`): List of object instanced from the class ``SliceStr``\
                 that defines the structure of an individual slice.

        Examples:
            >>> from numpy import array
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.slipsurface import CircularSurface
            >>> from pybimstab.slices import MaterialParameters, Slices
            >>> slope = AnthropicSlope(slopeHeight=7.5, slopeDip=[1, 1.5],
            >>>                        crownDist=5, toeDist=5)
            >>> surface = CircularSurface(slopeCoords=slope.coords,
            >>>                           dist1=2, dist2=10, radius=9)
            >>> material = MaterialParameters(
            >>>     cohesion=15, frictAngle=23, unitWeight=17,
            >>>     blocksUnitWeight=21, wtUnitWeight=9.8)
            >>> slices = Slices(
            >>>     material=material, slipSurfCoords=surface.coords,
            >>>     slopeCoords=slope.coords, numSlices=10,
            >>>     watertabCoords=None, bim=None)
            >>> slicesList = slices.createSlices()
            >>> slicesList[0].__dict__.keys()
            dict_keys(['material', 'terrainLS', 'slipSurfLS', 'watertabLS',
                       'bim', 'terrainCoords', 'slipSurfCoords',
                       'watertabCoords', 'sliceLS', 'coords', 'xMin', 'xMax',
                       'yMin', 'yMax', 'area', 'width', 'baseSlope', 'alpha',
                       'baseLength', 'l', 'topLength', 'topSlope',
                       'topInclinatDeg', 'midHeight', 'midWatTabHeight'])
        '''
        import numpy as np
        from shapely.geometry import LineString

        from pybimstab.tools import getPointAtX, extractSegment

        # Defining horizontal coordinates of the boundaries between slices
        maxHztSlipDist = self.slipSurfCoords[0, -1]
        minHztSlipDist = self.slipSurfCoords[0, 0]
        xCoords = np.linspace(minHztSlipDist, maxHztSlipDist, self.numSlices+1)
        if self.bim is not None:
            sliceWidth = (maxHztSlipDist - minHztSlipDist) / self.numSlices
            k = np.ceil(sliceWidth / self.bim.tileSize)
            xCoords = np.arange(minHztSlipDist,
                                maxHztSlipDist + k * self.bim.tileSize,
                                k * self.bim.tileSize)
            xCoords = list(xCoords)
            while xCoords[-1] > maxHztSlipDist:
                xCoords.pop()
            xCoords.append(maxHztSlipDist)
            xCoords = np.unique(xCoords)
            setattr(self, 'numSlices', len(xCoords) - 1)

        # Transform the  surfaces to LineString
        terrainLS = LineString(self.slopeCoords[:, 1:-2].T)
        slipSurfLS = LineString(self.slipSurfCoords.T)
        if self.watertabCoords is not None:
            waterTabLS = LineString(self.watertabCoords.T)

        # Getting the segments and defining the slices' structure
        slices = list()
        for n in range(self.numSlices):
            # Extracting the segment from the terrain surface
            pt1 = getPointAtX(terrainLS, xCoords[n])
            pt2 = getPointAtX(terrainLS, xCoords[n+1])
            terrainSegLS = extractSegment(
                terrainLS, terrainLS.project(pt1), terrainLS.project(pt2))
            # Extracting the segment from the slip surface
            pt1 = getPointAtX(slipSurfLS, xCoords[n])
            pt2 = getPointAtX(slipSurfLS, xCoords[n+1])
            slipSurfSegLS = extractSegment(
                slipSurfLS, slipSurfLS.project(pt1), slipSurfLS.project(pt2))
            # Getting the ns from the water table
            if self.watertabCoords is not None:
                pt1 = getPointAtX(waterTabLS, xCoords[n])
                pt2 = getPointAtX(waterTabLS, xCoords[n+1])
                waterTabSegLS = extractSegment(waterTabLS,
                                               waterTabLS.project(pt1),
                                               waterTabLS.project(pt2))
            else:
                waterTabSegLS = None
            # Define the structure of the individual slice and append to list
            slices.append(
                SliceStr(material=self.material, terrainLS=terrainSegLS,
                         slipSurfLS=slipSurfSegLS, watertabLS=waterTabSegLS,
                         bim=self.bim))
        setattr(self, 'slices', slices)
        return slices

    def setExtLoads(self, extL=[{'load': 0, 'angle': 0}]):
        '''Method for setting the external loads to the slices.

        Args:
            extL (`list`): list that stores the information of the external
                loads in dictionaries. Each dictionary has the value of the
                external force (at the slope surface) and its inclination in
                degrees as the following structure ``{'load': 0, 'angle': 0}``.
                There are three possibilities to define the structure:

                    - Unitary list means that the input is constant for all \
                        the slices. ``{'load': 0, 'angle': 0}`` is the default\
                        value.
                    - If the length of the list is as long as the number of\
                        slices, each slice is coupled with the input list.
                    - If the length of the list is :math:`l>1` and lower than\
                        the number of slices, then only the :math:`l` fisrt\
                        slices are coupled with the external loads.

        Examples:
            >>> from numpy import array
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.slipsurface import CircularSurface
            >>> from pybimstab.slices import MaterialParameters, Slices
            >>> slope = AnthropicSlope(slopeHeight=7.5, slopeDip=[1, 1.5],
            >>>                        crownDist=5, toeDist=5)
            >>> surface = CircularSurface(slopeCoords=slope.coords,
            >>>                           dist1=2, dist2=10, radius=9)
            >>> material = MaterialParameters(
            >>>     cohesion=15, frictAngle=23, unitWeight=17,
            >>>     blocksUnitWeight=21, wtUnitWeight=9.8)
            >>> slices = Slices(
            >>>     material=material, slipSurfCoords=surface.coords,
            >>>     slopeCoords=slope.coords, numSlices=10,
            >>>     watertabCoords=None, bim=None)
            >>> slices.setExtLoads(extL=[{'load': 200, 'angle': 30}])
            >>> slices.slices[0].extL, slices.slices[0].w
            (200, 30)
        '''
        # Setting the external loads to each slice as a new attribute
        for i in range(self.numSlices):
            slice_ = self.slices[i]
            if len(extL) or i > len(extL):
                slice_.extL = extL[0]['load']
                slice_.w = extL[0]['angle']
            else:
                slice_.extL = extL[i]['load']
                slice_.w = extL[i]['angle']
        return

    def plot(self, plotFittedCirc=False):
        '''Method for generating a graphic of the slope stability model when is
        possible to watch the slices in the soil mass above the slip surface.

        Returns:
            (`matplotlib.figure.Figure`): object with the matplotlib structure\
                of the plot. You might use it to save the figure for example.

        Examples:
            >>> from numpy import array
            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.slipsurface import CircularSurface
            >>> from pybimstab.slices import MaterialParameters, Slices
            >>> slope = AnthropicSlope(slopeHeight=7.5, slopeDip=[1, 1.5],
            >>>                        crownDist=5, toeDist=5)
            >>> surface = CircularSurface(slopeCoords=slope.coords,
            >>>                           dist1=2, dist2=10, radius=9)
            >>> material = MaterialParameters(
            >>>     cohesion=15, frictAngle=23, unitWeight=17,
            >>>     blocksUnitWeight=21, wtUnitWeight=9.8)
            >>> slices = Slices(
            >>>     material=material, slipSurfCoords=surface.coords,
            >>>     slopeCoords=slope.coords, numSlices=10,
            >>>     watertabCoords=None, bim=None)
            >>> fig = slices.plot()

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/slices_Slices_example1.svg
                :alt: slices_Slices_example1

            .. only:: html

                :download:`example script<../examples/figuresScripts/slices_Slices_example1.py>`.

            >>> from numpy import array
            >>> from pybimstab.slope import NaturalSlope
            >>> from pybimstab.watertable import WaterTable
            >>> from pybimstab.bim import BlocksInMatrix
            >>> from pybimstab.slipsurface import CircularSurface
            >>> from pybimstab.slipsurface import TortuousSurface
            >>> from pybimstab.slices import MaterialParameters, Slices
            >>> terrainCoords = array(
            >>>     [[-2.49, 0.1, 1.7, 3.89, 5.9, 8.12, 9.87, 13.29, 20.29,
            >>>       21.43, 22.28, 23.48, 24.65, 25.17],
            >>>      [18.16, 17.88, 17.28, 15.73, 14.31, 13.58, 13, 3.61, 3.61,
            >>>       3.32, 2.71, 2.23, 1.21, 0.25]])
            >>> slope = NaturalSlope(terrainCoords)
            >>> bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.3,
            >>>                      tileSize=0.35, seed=123)
            >>> watertabDepths = array([[0, 5, 10, 15],
            >>>                         [8, 7, 3, 0]])
            >>> watertable = WaterTable(slopeCoords=slope.coords,
            >>>                         watertabDepths=watertabDepths,
            >>>                         smoothFactor=3)
            >>> preferredPath = CircularSurface(
            >>>     slopeCoords=slope.coords, dist1=5, dist2=15.78, radius=20)
            >>> surface = TortuousSurface(
            >>>     bim, dist1=4, dist2=15.78, heuristic='euclidean',
            >>>     reverseLeft=False, reverseUp=False, smoothFactor=2,
            >>>     preferredPath=preferredPath.coords, prefPathFact=2)
            >>> material = MaterialParameters(
            >>>     cohesion=15, frictAngle=23, unitWeight=17,
            >>>     blocksUnitWeight=21, wtUnitWeight=9.8)
            >>> slices = Slices(
            >>>     material=material, slipSurfCoords=surface.coords,
            >>>     slopeCoords=slope.coords, numSlices=10,
            >>>     watertabCoords=watertable.coords, bim=bim)
            >>> fig = slices.plot()

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/slices_Slices_example2.svg
                :alt: slices_Slices_example2

            .. only:: html

                :download:`example script<../examples/figuresScripts/slices_Slices_example2.py>`.
        '''
        import numpy as np
        from matplotlib import pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap as newcmap
        from pybimstab.slipsurface import CircularSurface

        # Variables to control the color map and its legend
        if self.bim is not None:
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
        if self.bim is not None:
            bar = ax.pcolormesh(self.bim.xCells, self.bim.yCells,
                                self.bim.grid, cmap=cmap)
            # Configuring the colorbar
            bar = plt.colorbar(bar, ax=ax, ticks=ticks, pad=0.05,
                               shrink=0.15, aspect=3)
            bar.ax.set_yticklabels(ticksLabels, fontsize='small')
        if plotFittedCirc:
            circSurf = CircularSurface(slopeCoords=self.slopeCoords,
                                       dist1=self.fittedCirc['dist1'],
                                       dist2=self.fittedCirc['dist2'],
                                       radius=self.fittedCirc['radius'])
            ax.plot(*circSurf.coords, ':r', label='Fitted cir. surf.')
        for slice_ in self.slices:
            ax.plot(slice_.coords[0], slice_.coords[1], ':r', lw=0.5)
        ax.plot(self.slipSurfCoords[0], self.slipSurfCoords[1], '-r',
                label='slip surface')
        ax.plot(self.slopeCoords[0], self.slopeCoords[1], '-k')
        if self.watertabCoords is not None:
            ax.plot(self.watertabCoords[0], self.watertabCoords[1],
                    'deepskyblue', lw=0.9, label='watertable')
        # Plot settings
        ax.set_aspect(1)
        ax.legend(fontsize='small', bbox_to_anchor=(1.005, 1), loc=2)
        ax.grid(True, ls='--', lw=0.5)
        ax.set_xlim((self.slopeCoords[0].min()-0.02*self.slopeCoords[0].max(),
                     1.02*self.slopeCoords[0].max()))
        ax.set_ylim((self.slopeCoords[1].min()-0.02*self.slopeCoords[1].max(),
                     1.02*self.slopeCoords[1].max()))
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
