from numpy.testing import assert_allclose

from astropy import units as u

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from poliastro.twobody.propagation import cowell

from argp import guidance_law, extra_quantities


def test_soyuz_standard_gto_delta_v():
    # Data from Soyuz Users Manual, issue 2 revision 0
    r_a = (Earth.R + 35950 * u.km).to(u.km).value
    r_p = (Earth.R + 250 * u.km).to(u.km).value

    a = (r_a + r_p) / 2  # km
    ecc = r_a / a - 1
    argp_0 = (178 * u.deg).to(u.rad).value  # rad
    argp_f = (178 * u.deg + 5 * u.deg).to(u.rad).value  # rad
    f = 2.4e-7  # km / s2

    k = Earth.k.decompose([u.km, u.s]).value

    expected_t_f = 12.0  # days, approximate
    expected_delta_V = 0.2489  # km / s

    delta_V, t_f = extra_quantities(k, a, ecc, argp_0, argp_f, f)

    assert_allclose(delta_V, expected_delta_V, rtol=1e-2)
    assert_allclose(t_f / 86400, expected_t_f, rtol=1e-2)


def test_soyuz_standard_gto_numerical():
    # Data from Soyuz Users Manual, issue 2 revision 0
    r_a = (Earth.R + 35950 * u.km).to(u.km).value
    r_p = (Earth.R + 250 * u.km).to(u.km).value

    a = (r_a + r_p) / 2  # km
    ecc = r_a / a - 1
    argp_0 = (178 * u.deg).to(u.rad).value  # rad
    argp_f = (178 * u.deg + 5 * u.deg).to(u.rad).value  # rad
    f = 2.4e-7  # km / s2

    k = Earth.k.decompose([u.km, u.s]).value

    optimal_accel = guidance_law(f)

    _, t_f = extra_quantities(k, a, ecc, argp_0, argp_f, f)

    # Retrieve r and v from initial orbit
    s0 = Orbit.from_classical(
        Earth,
        a * u.km, (r_a / a - 1) * u.one, 6 * u.deg,
        188.5 * u.deg, 178 * u.deg, 0 * u.deg
    )
    r0, v0 = s0.rv()

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

    assert_allclose(sf.argp.to(u.rad).value, argp_f, rtol=1e-4)
