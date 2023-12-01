from shapely.geometry import LineString
from pybimstab.slope import AnthropicSlope
from pybimstab.bim import BlocksInMatrix
from pybimstab.slices import MaterialParameters, SliceStr
material = MaterialParameters(cohesion=15, frictAngle=23,
                              unitWeight=17)
slope = AnthropicSlope(slopeHeight=7.5, slopeDip=[1, 1.5],
                       crownDist=5, toeDist=5, depth=2)
bim = BlocksInMatrix(slopeCoords=slope.coords, blockProp=0.2,
                     tileSize=0.5, seed=123)
terrainLS = LineString([(6, 8), (7, 6.5)])
slipSurfLS = LineString([(6, 3.395), (7, 2.837)])
watertabLS = LineString([(6, 5), (7, 4)])
slice_ = SliceStr(material, terrainLS, slipSurfLS, watertabLS,
                  bim=bim)
fig = slice_.plot()
