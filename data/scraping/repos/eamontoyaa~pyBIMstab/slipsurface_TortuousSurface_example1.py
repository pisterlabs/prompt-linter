from pybimstab.slope import AnthropicSlope
from pybimstab.bim import BlocksInMatrix
from pybimstab.slipsurface import TortuousSurface
slope = AnthropicSlope(slopeHeight=12, slopeDip=[1, 1.5],
                       crownDist=10, toeDist=10)
bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.25,
                     tileSize=0.25, seed=123)

# Not allowing to turn left and up
surface = TortuousSurface(
    bim, dist1=0, dist2=17, heuristic='manhattan',
    reverseLeft=False, reverseUp=False, smoothFactor=0,
    preferredPath=None, prefPathFact=None)
fig = surface.plot()
