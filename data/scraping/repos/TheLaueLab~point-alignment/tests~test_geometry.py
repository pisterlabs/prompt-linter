import numpy as np
import pytest
from itertools import islice, starmap

from coherent_point_drift.geometry import *

def test_random_rotations():
    np.testing.assert_equal(next(randomRotations(3, 4)), next(randomRotations(3, 4)))
    with pytest.raises(AssertionError):
        np.testing.assert_equal(*islice(randomRotations(3), 2))
    with pytest.raises(AssertionError):
        np.testing.assert_equal(next(randomRotations(3)), next(randomRotations(3)))

def test_rmsd():
    rng = np.random.RandomState(4)
    points = rng.uniform(0, 1, (10, 2))
    P = np.eye(len(points))

    assert RMSD(points, points + 1, P) > 0.5
    assert RMSD(points, points, P) == 0

def test_std():
    assert std(np.zeros((5, 3))) == 0
    unit = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    unit = np.concatenate([unit, -unit])
    assert std(unit) == 1

def test_rigid_associative():
    rng = np.random.RandomState(4)
    Rs = starmap(rotationMatrix, randomRotations(3, rng))
    xform1 = RigidXform(next(Rs), rng.normal(size=3), rng.lognormal(0.0, 0.5))
    xform2 = RigidXform(next(Rs), rng.normal(size=3), rng.lognormal(0.0, 0.5))

    points = rng.normal(size=(10, 3))
    np.testing.assert_allclose((xform1 @ xform2) @ points, xform1 @ (xform2 @ points))

def test_rigid_identity():
    rng = np.random.RandomState(4)
    xform = RigidXform(np.eye(3), np.zeros(3), 1)

    points = rng.normal(size=(10, 3))
    np.testing.assert_allclose(xform @ points, points)
    np.testing.assert_allclose(RigidXform() @ points, points)
    np.testing.assert_allclose(xform @ RigidXform() @ points, points)
    np.testing.assert_allclose(RigidXform() @ RigidXform() @ points, points)

def test_rigid_inverse():
    rng = np.random.RandomState(4)
    points = rng.normal(size=(10, 3))
    Rs = starmap(rotationMatrix, randomRotations(3, rng))

    xforms = [
        RigidXform(next(Rs), rng.normal(size=3), rng.lognormal(0.0, 0.5)),
        RigidXform(next(Rs)), RigidXform(t=rng.normal(size=3)), RigidXform()
    ]
    for xform in xforms:
        np.testing.assert_allclose(xform.inverse @ xform @ points, points)

def test_affine_associative():
    rng = np.random.RandomState(4)
    xform1 = AffineXform(rng.normal(size=(3, 3)), rng.normal(size=3))
    xform2 = AffineXform(rng.normal(size=(3, 3)), rng.normal(size=3))

    points = rng.normal(size=(10, 3))
    np.testing.assert_allclose((xform1 @ xform2) @ points, xform1 @ (xform2 @ points))

def test_affine_identity():
    rng = np.random.RandomState(4)
    xform = AffineXform(np.eye(3), np.zeros(3))

    points = rng.normal(size=(10, 3))
    np.testing.assert_allclose(xform @ points, points)
    np.testing.assert_allclose(AffineXform() @ points, points)
    np.testing.assert_allclose(xform @ AffineXform() @ points, points)
    np.testing.assert_allclose(AffineXform() @ AffineXform() @ points, points)

def test_affine_inverse():
    rng = np.random.RandomState(4)
    points = rng.normal(size=(10, 3))

    xforms = [
        AffineXform(rng.normal(size=(3, 3)), rng.normal(size=3)),
        AffineXform(rng.normal(size=(3, 3))),
        AffineXform(t=rng.normal(size=3)), AffineXform(),
    ]
    for xform in xforms:
        np.testing.assert_allclose(xform.inverse @ xform @ points, points)

def test_normalize():
    rng = np.random.RandomState(4)
    points = rng.uniform(-200, 200, size=(20, 2)) + np.array([[40, -2]])

    norm = AffineXform.normalize(points)
    np.testing.assert_allclose((norm @ points).mean(axis=0), 0, atol=1e-16)
    np.testing.assert_allclose(std(norm @ points), 1.0)

    norm = RigidXform.normalize(points)
    np.testing.assert_allclose((norm @ points).mean(axis=0), 0, atol=1e-16)
    np.testing.assert_allclose(std(norm @ points), 1.0)
