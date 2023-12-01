from pybimstab.slope import AnthropicSlope
from pybimstab.bim import BlocksInMatrix
slope = AnthropicSlope(slopeHeight=12, slopeDip=[1, 1.5],
                       crownDist=10, toeDist=10)
bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.2,
                     tileSize=0.25, seed=123)
fig = bim.plot()
