from numpy import array
from pybimstab.slope import AnthropicSlope
from pybimstab.watertable import WaterTable
slope = AnthropicSlope(slopeHeight=7.5, slopeDip=[1, 1.5],
                       crownDist=5, toeDist=5)
watertabDepths = array([[0, 2, 5, 7, 12, 15],
                       [2.5, 2.5, 3, 1.5, 0.5, 1]])
watertable = WaterTable(slopeCoords=slope.coords,
                        watertabDepths=watertabDepths,
                        smoothFactor=0)
fig = watertable.plot()
