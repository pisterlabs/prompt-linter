from pybimstab.slope import AnthropicSlope
from pybimstab.slipsurface import CircularSurface
from pybimstab.slices import MaterialParameters, Slices
slope = AnthropicSlope(slopeHeight=7.5, slopeDip=[1, 1.5],
                       crownDist=5, toeDist=5)
surface = CircularSurface(slopeCoords=slope.coords,
                          dist1=2, dist2=10, radius=9)
material = MaterialParameters(
    cohesion=15, frictAngle=23, unitWeight=17,
    blocksUnitWeight=21, wtUnitWeight=9.8)
slices = Slices(
    material=material, slipSurfCoords=surface.coords,
    slopeCoords=slope.coords, numSlices=10,
    watertabCoords=None, bim=None)
fig = slices.plot()
