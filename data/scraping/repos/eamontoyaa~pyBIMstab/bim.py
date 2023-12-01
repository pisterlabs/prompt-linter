# -*- coding: utf-8 -*-
'''
Module for defining the class related the structure of the BIM.
'''


# %%
class BlocksInMatrix:
    '''Creates an instance of an object that defines the structure of the
    block-in-rock material (BIM) that composes the slope. ::

        BlocksInMatrix(slopeCoords, blockProp, tileSize, seed=None)

    The BIM is defined as a grided array of vertical square tiles. Each tile
    is composed either of block or matrix. Those cells that correspond to a
    block-tile have value one (1), those cells that correspond to a
    matrix-tile have value zero (0), and those cells tath are located outside
    the polygon have vale minus one (-1).

    Attributes:
        slopeCoords ((n, 2) `numpy.ndarray`): Coordinates of the polygon within
            which the BIM is defined. It is expected that the polygon
            corresponds to the slope boundary that is obtained with the method
            ``defineBoundary`` either from the classes ``AnthropicSlope`` or
            ``NaturalSlope`` (module ``slope``), however it works for any
            closed polygon.
        blockProp (`float`): Proportion of blocks relative to the total
            volume of the BIM. It's given as a value between 0 and 1.
        tileSize (``int`` or ``float``): Length of each tile-side.
        seed (``float``): Seed value for repeatability in the random generation
            of the blocks.

    Note:
        The class ``BlocksInMatrix`` requires `numpy <http://www.numpy.org/>`_
        and `matplotlib <https://matplotlib.org/>`_.

    Examples:
        >>> from pybimstab.slope import AnthropicSlope
        >>> from pybimstab.bim import BlocksInMatrix
        >>> slope = AnthropicSlope(slopeHeight=12, slopeDip=[1, 1.5],
        >>>                        crownDist=10, toeDist=10)
        >>> bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.25,
        >>>                      tileSize=0.25)
        >>> bim.__dict__.keys()
        dict_keys(['slopeCoords', 'blockProp', 'tileSize', 'seed', 'grid',
                   'xCells', 'yCells'])
        '''

    def __init__(self, slopeCoords, blockProp, tileSize, seed=None):
        '''
        BlocksInMatrix(slopeCoords, blockProp, tileSize, seed=None)
        '''
        self.slopeCoords = slopeCoords
        self.blockProp = blockProp
        self.tileSize = tileSize
        self.seed = seed
        # Create the BIM structure
        self.defineGrid()

    def defineGrid(self):
        '''Method to create the grid-structure of the BIM into de boundary.

        Those cells that correspond to a block-tile have value one (1), those
        cells that correspond to a matrix-tile have value zero (0), and those
        cells tath are located outside the polygon have vale minus one (-1).

        Returns:
            (`numpy.ndarray`): :math:`\\left(m \\times n \\right)` matrix\
                that defines a grid-graph that represents the structure of the\
                BIM.

        Examples:

            >>> from numpy import array
            >>> from pybimstab.bim import BlocksInMatrix
            >>> slopeCoords = array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
            >>> bim = BlocksInMatrix(slopeCoords=slopeCoords, blockProp=0.5,
            >>>                      tileSize=0.1, seed=123)
            >>> bim.defineGrid()
            array([[1., 0., 0., 1., 1., 0., 1., 1., 0., 0.],
                   [0., 1., 0., 0., 0., 1., 0., 0., 1., 1.],
                   [1., 1., 1., 1., 1., 0., 0., 0., 0., 1.],
                   [0., 0., 0., 0., 0., 0., 0., 1., 1., 1.],
                   [1., 0., 0., 0., 1., 0., 0., 1., 1., 1.],
                   [0., 1., 1., 1., 0., 0., 0., 1., 1., 1.],
                   [1., 1., 1., 1., 1., 0., 1., 0., 0., 1.],
                   [0., 1., 1., 1., 0., 1., 1., 0., 0., 1.],
                   [0., 1., 1., 0., 1., 1., 0., 0., 0., 0.]])

            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.bim import BlocksInMatrix
            >>> slope = AnthropicSlope(slopeHeight=12, slopeDip=[1, 1.5],
            >>>                        crownDist=10, toeDist=10)
            >>> bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.25,
            >>>                      tileSize=0.25, seed=123)
            >>> bim.defineGrid()
            array([[-1., -1., -1., ..., -1., -1., -1.],
                   [ 0.,  1.,  0., ...,  1.,  0.,  1.],
                   [ 1.,  0.,  0., ...,  1.,  0.,  0.],
                   ...,
                   [ 0.,  1.,  0., ..., -1., -1., -1.],
                   [ 0.,  0.,  0., ..., -1., -1., -1.],
                   [ 0.,  0.,  0., ..., -1., -1., -1.]])
        '''
        import numpy as np

        from pybimstab.polygon import Polygon

        def defGrid(tileSize, blockProp):
            # Meshgrid for the lower-left corner of the tiles
            yCells, xCells = np.mgrid[
                    slice(self.slopeCoords[1].min(),
                          self.slopeCoords[1].max()+tileSize, tileSize),
                    slice(self.slopeCoords[0].min(),
                          self.slopeCoords[0].max()+tileSize, tileSize)]
            gridDim = np.array(xCells.shape) - 1  # dimension of the grid
            # difference of heights and displacement of the meshgrid
            gridHeight = gridDim[0] * tileSize
            slopeHeight = self.slopeCoords[1].max() - self.slopeCoords[1].min()
            dy = gridHeight - slopeHeight  # excedent at the top of the polygon
            yCells -= dy

            # Pixels that are out of the slope boundary.
            polygon = Polygon(self.slopeCoords)
            inSlope = polygon.isinside(xCells[0, :-1] + 0.5 * tileSize,
                                       yCells[:-1, 0] + 0.5 * tileSize,
                                       meshgrid=True, want2plot=False)
            outSlopeIdx = np.where(inSlope == 0)

            # Creating the blocks from a random binomial distribution
            np.random.seed(self.seed)  # defining the seed for repeatability
            grid = np.random.binomial(1, blockProp, gridDim)
            grid[outSlopeIdx] = -1  # Outer has cost -1
            return grid, xCells, yCells, outSlopeIdx

        grid, xCells, yCells, outSlopeIdx = defGrid(
                self.tileSize, self.blockProp)

        gridAux, xCellsAux, yCellsAux, outSlopeIdxAux = defGrid(
                self.tileSize * 0.2, 0)

        # Setting the attribute to the instanced object.
        setattr(self, 'grid', grid)
        setattr(self, 'xCells', xCells)
        setattr(self, 'yCells', yCells)
        setattr(self, 'outSlopeIdx', outSlopeIdx)
        setattr(self, 'gridAux', gridAux)
        setattr(self, 'xCellsAux', xCellsAux)
        setattr(self, 'yCellsAux', yCellsAux)
        setattr(self, 'outSlopeIdxAux', outSlopeIdxAux)
        return grid

    def plot(self):
        '''Method for generating a graphic of the grid structure of the BIM.

        Returns:
            (`matplotlib.figure.Figure`): object with the matplotlib structure\
                of the plot. You might use it to save the figure for example.

        Examples:
            >>> from numpy import array
            >>> from pybimstab.bim import BlocksInMatrix
            >>> slopeCoords = array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
            >>> bim = BlocksInMatrix(slopeCoords=slopeCoords, blockProp=0.5,
            >>>                      tileSize=0.1, seed=123)
            >>> fig = bim.plot()

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/bim_example1.svg
                :alt: bim_example1

            .. only:: html

               :download:`example script<../examples/figuresScripts/bim_example1.py>`.

            >>> from pybimstab.slope import AnthropicSlope
            >>> from pybimstab.bim import BlocksInMatrix
            >>> slope = AnthropicSlope(slopeHeight=12, slopeDip=[1, 1.5],
            >>>                        crownDist=10, toeDist=10)
            >>> bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.2,
            >>>                      tileSize=0.25, seed=123)
            >>> fig = bim.plot()

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/bim_example2.svg
                :alt: bim_example2

            .. only:: html

               :download:`example script<../examples/figuresScripts/bim_example2.py>`.
                        fig = bim.plot()

            >>> from numpy import array
            >>> from pybimstab.slope import NaturalSlope
            >>> from pybimstab.bim import BlocksInMatrix
            >>> terrainCoords = array(
            >>>     [[-2.49, 0.1 , 1.7 , 3.89, 5.9 , 8.12, 9.87, 13.29, 20.29,
            >>>       21.43, 22.28, 23.48, 24.65, 25.17],
            >>>      [18.16, 17.88, 17.28, 15.73, 14.31, 13.58, 13, 3.61, 3.61,
            >>>       3.32, 2.71, 2.23, 1.21, 0.25]])
            >>> slope = NaturalSlope(terrainCoords)
            >>> bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.25,
            >>>                      tileSize=0.4, seed=123)
            >>> fig = bim.plot()

            .. figure:: https://rawgit.com/eamontoyaa/pybimstab/master/examples/figures/bim_example3.svg
                :alt: bim_example3

            .. only:: html

               :download:`example script<../examples/figuresScripts/bim_example3.py>`.
        '''
        import numpy as np
        from matplotlib import pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap as newcmap

        # Variables to control the color map and its legend
        if np.any(self.grid == -1):
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
        bar = ax.pcolormesh(self.xCells, self.yCells, self.grid, cmap=cmap)
        ax.plot(self.slopeCoords[0], self.slopeCoords[1], '-k')
        # Configuring the colorbar
        bar = plt.colorbar(bar, ax=ax, ticks=ticks, pad=0.005,
                           shrink=0.15, aspect=3)
        bar.ax.set_yticklabels(ticksLabels, fontsize='small')
        # Plot settings
        ax.set_aspect(1)
        ax.grid(False, ls='--', lw=0)
#        ax.grid(True, ls='--', lw=0.5)
#        ax.grid(True, ls='--', lw=0.5)
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
