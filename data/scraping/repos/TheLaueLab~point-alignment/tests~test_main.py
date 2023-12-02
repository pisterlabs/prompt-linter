import pytest
import numpy as np

from coherent_point_drift.main import *

def test_print_points(tmpdir):
    points = np.asarray([
        [1.0, -5.0, 3.0],
        [2.0, 0.0, 0.0],
    ])

    filename = tmpdir.join("tmp")
    with filename.open("wb") as f:
        savePoints(f, points, 'txt')
    with filename.open("rb") as f:
        txt = f.read().decode()

    assert txt == (
        "1.0 -5.0 3.0\n"
        "2.0 0.0 0.0\n"
    )


def test_print_xform(tmpdir):
    xform = RigidXform(np.asarray([[1.0, 0.3], [0.3, 2.0]]), np.asarray([0.4, 1.0]), 5.0)

    filename = tmpdir.join("tmp")
    with filename.open("wb") as f:
        saveXform(f, xform, 'print')
    with filename.open("rb") as f:
        txt = f.read().decode()

    assert txt == (
        "1.0 0.3\n"
        "0.3 2.0\n"
        "\n"
        "0.4 1.0\n"
        "\n"
        "5.0\n"
    )
