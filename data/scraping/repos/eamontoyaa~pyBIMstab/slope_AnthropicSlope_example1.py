from pybimstab.slope import AnthropicSlope
slope = AnthropicSlope(slopeHeight=12, slopeDip=[1, 1.5],
                       crownDist=10, toeDist=10)
fig = slope.plot()
