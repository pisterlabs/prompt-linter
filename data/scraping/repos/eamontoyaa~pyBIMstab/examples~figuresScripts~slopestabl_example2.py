# Example Case 5 (Fredlund & Krahn, 1977)
from numpy import array
from pybimstab.slope import AnthropicSlope
from pybimstab.slipsurface import CircularSurface
from pybimstab.watertable import WaterTable
from pybimstab.slices import MaterialParameters, Slices
from pybimstab.slopestabl import SlopeStabl
slope = AnthropicSlope(slopeHeight=40, slopeDip=[2, 1],
                       crownDist=60, toeDist=30, depth=20)
surface = CircularSurface(slopeCoords=slope.coords,
                          dist1=45.838, dist2=158.726,
                          radius=80)
material = MaterialParameters(cohesion=600, frictAngle=20,
                              unitWeight=120,
                              wtUnitWeight=62.4)
watertable = WaterTable(slopeCoords=slope.coords,
                        watertabDepths=array([[0, 140],
                                              [20, 0]]))
slices = Slices(
    material=material, slipSurfCoords=surface.coords,
    slopeCoords=slope.coords, numSlices=50,
    watertabCoords=watertable.coords, bim=None)
stabAnalysis = SlopeStabl(slices, seedFS=1, Kh=0)
fig = stabAnalysis.plot()
