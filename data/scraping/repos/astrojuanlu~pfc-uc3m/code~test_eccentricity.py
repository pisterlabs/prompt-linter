import pytest

from numpy.testing import assert_allclose

from astropy import units as u

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from poliastro.twobody.propagation import cowell

from eccentricity_quasioptimal import guidance_law, extra_quantities


@pytest.mark.parametrize("ecc_0,ecc_f", [
    [0.0, 0.1245],  # Reverse-engineered from results
    [0.1245, 0.0]
])
def test_sso_disposal_time_and_delta_v(ecc_0, ecc_f):
    a_0 = Earth.R.to(u.km).value + 900  # km
    f = 2.4e-7  # km / s2, assumed constant

    k = Earth.k.decompose([u.km, u.s]).value

    expected_t_f = 29.697  # days, reverse-engineered
    expected_delta_V = 0.6158  # km / s, lower than actual result

    delta_V, t_f = extra_quantities(k, a_0, ecc_0, ecc_f, f)

    assert_allclose(delta_V, expected_delta_V, rtol=1e-4)
    assert_allclose(t_f / 86400, expected_t_f, rtol=1e-4)


@pytest.mark.parametrize("ecc_0,ecc_f", [
    [0.0, 0.1245],  # Reverse-engineered from results
    [0.1245, 0.0]
])
def test_sso_disposal_numerical(ecc_0, ecc_f):
    a_0 = Earth.R.to(u.km).value + 900  # km
    f = 2.4e-7  # km / s2, assumed constant

    k = Earth.k.decompose([u.km, u.s]).value

    _, t_f = extra_quantities(k, a_0, ecc_0, ecc_f, f)

    # Retrieve r and v from initial orbit
    s0 = Orbit.from_classical(Earth, a_0 * u.km, ecc_0 * u.one,
                              0 * u.deg, 0 * u.deg, 0 * u.deg, 0 * u.deg)
    r0, v0 = s0.rv()

    optimal_accel = guidance_law(s0, ecc_f, f)

    # Propagate orbit
    r, v = cowell(k,
                  r0.to(u.km).value,
                  v0.to(u.km / u.s).value,
                  t_f,
                  ad=optimal_accel,
                  nsteps=1000000)

    sf = Orbit.from_vectors(Earth,
                            r * u.km,
                            v * u.km / u.s,
                            s0.epoch + t_f * u.s)

    assert_allclose(sf.ecc.value, ecc_f, rtol=1e-4, atol=1e-4)
