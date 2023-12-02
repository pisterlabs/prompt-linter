import pytest

import numpy as np
from numpy.testing import assert_allclose

from astropy import units as u

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from poliastro.twobody.propagation import cowell

from edelbaum import guidance_law, extra_quantities

# Problem data
f = 3.5e-7  # km / s2

a_0 = 7000.0  # km
a_f = 42166.0  # km
inc_f = 0.0  # rad

k = Earth.k.decompose([u.km, u.s]).value


@pytest.mark.parametrize("inc_0,expected_t_f,expected_delta_V,rtol", [
    [28.5, 191.26295, 5.78378, 1e-5],
    [90.0, 335.0, 10.13, 1e-3],  # Extra decimal places added
    [114.591, 351.0, 10.61, 1e-2]
])
def test_leo_geo_time_and_delta_v(inc_0, expected_t_f, expected_delta_V, rtol):
    inc_0 = np.radians(inc_0)  # rad

    delta_V, t_f = extra_quantities(k, a_0, a_f, inc_0, inc_f, f)

    assert_allclose(delta_V, expected_delta_V, rtol=rtol)
    assert_allclose(t_f / 86400, expected_t_f, rtol=rtol)


@pytest.mark.parametrize("inc_0", [np.radians(28.5), np.radians(90.0)])
def test_leo_geo_numerical(inc_0):
    edelbaum_accel = guidance_law(k, a_0, a_f, inc_0, inc_f, f)

    _, t_f = extra_quantities(k, a_0, a_f, inc_0, inc_f, f)

    # Retrieve r and v from initial orbit
    s0 = Orbit.circular(Earth, a_0 * u.km - Earth.R, inc_0 * u.rad)
    r0, v0 = s0.rv()

    # Propagate orbit
    r, v = cowell(k,
                  r0.to(u.km).value,
                  v0.to(u.km / u.s).value,
                  t_f,
                  ad=edelbaum_accel,
                  nsteps=1000000)

    sf = Orbit.from_vectors(Earth,
                            r * u.km,
                            v * u.km / u.s,
                            s0.epoch + t_f * u.s)

    assert_allclose(sf.a.to(u.km).value, a_f, rtol=1e-5)
    assert_allclose(sf.ecc.value, 0.0, atol=1e-2)
    assert_allclose(sf.inc.to(u.rad).value, inc_f, atol=1e-3)
