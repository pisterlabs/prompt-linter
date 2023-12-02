from pybimstab.slope import AnthropicSlope
from pybimstab.slipsurface import CircularSurface
slope = AnthropicSlope(slopeHeight=7.5, slopeDip=[1, 1.5],
                       crownDist=5, toeDist=5)
surface = CircularSurface(slopeCoords=slope.coords,
                          dist1=2, dist2=10, radius=9)
fig = surface.plot()
