from itertools import islice, product as cartesian, starmap
from coherent_point_drift.geometry import rigidXform, rotationMatrix, RMSD, randomRotations
from coherent_point_drift.align import globalAlignment
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

fig, axs = plt.subplots(2, 1)

for ndim, ax in zip((2, 3), axs):
    ax.set_title("{}D".format(ndim))
    nstepss = range(1, 8)
    pointss = np.random.random((10, 12, ndim))
    rotations = starmap(rotationMatrix, islice(randomRotations(ndim), 10))

    rmsds = defaultdict(list)
    for nsteps, points, rotation in cartesian(nstepss, pointss, rotations):
        degraded = rigidXform(points, R=rotation)
        xform = globalAlignment(points, degraded, w=0.1, nsteps=nsteps)
        rmsds[nsteps].append(RMSD(points, rigidXform(degraded, *xform)))
    labels = sorted(rmsds.keys())
    ax.violinplot([rmsds[i] for i in labels], labels)
fig.tight_layout()
plt.show()
