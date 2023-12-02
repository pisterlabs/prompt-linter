import os
from datetime import datetime

import numpy as np
from numpy.linalg import norm

from matplotlib import rc
import matplotlib.pyplot as plt

from astropy import units as u

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from poliastro.twobody.propagation import cowell
from poliastro.twobody.rv import RVState

from combined_ei import guidance_law, extra_quantities

from util import ProgressResultsCallback


def _compute_results_array(a, ecc_0, ecc_f, inc_0, inc_f, argp, f):
    k = Earth.k.decompose([u.km, u.s]).value

    _, _, t_f = extra_quantities(k, a, ecc_0, ecc_f, inc_0, inc_f, argp, f)

    # Retrieve r and v from initial orbit
    s0 = Orbit.from_classical(
        Earth,
        a * u.km, ecc_0 * u.one, inc_0 * u.deg,
        0 * u.deg, argp * u.deg, 0 * u.deg
    )
    r0, v0 = s0.rv()

    combined_ei_accel = guidance_law(s0, ecc_f, inc_f, f)

    # Propagate orbit
    with ProgressResultsCallback(t_f) as res:
        cowell(
            k, r0.to(u.km).value, v0.to(u.km / u.s).value, t_f,
            ad=combined_ei_accel,
            callback=res,
            nsteps=1000000
        )

    return res.get_results()  # ~70 k rows, 7 columns, 3 MB in memory


def _extract_arrays(t_domain, r_vectors, v_vectors):
    ecc_values = np.zeros_like(t_domain)
    for ii in range(len(t_domain)):
        r = r_vectors[ii]
        v = v_vectors[ii]
        if r.any():
            ss = RVState(Earth, r * u.km, v * u.km / u.s)
            ecc_values[ii] = ss.ecc.value

    return ecc_values


def _plot_quantities(t_domain, ecc_values):
    # http://matplotlib.org/users/pgf.html#custom-preamble
    # http://sbillaudelle.de/2015/02/23/seamlessly-embedding-matplotlib-output-into-latex.html
    rc("pgf", rcfonts=False)
    rc("text", usetex=True)

    fig, ax = plt.subplots()
    ax.set_xlabel("Time, days")

    ax.plot(t_domain / 86400, ecc_values, color='k')
    ax.set_ylabel("Eccentricity")

    return fig


def _save_data(t_, ecc_):
    dir_name = "combined_ei_{}_{}".format(
        int(np.degrees(ecc_[0])), datetime.now().strftime("%m%d_%H_%M_%S"))

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    np.savetxt(os.path.join(dir_name, "data.txt"), np.column_stack([t_, ecc_]))
    return dir_name


def plot_geo_case(ecc_0, ecc_f):
    a = 42164  # km
    ecc_0 = 0.4
    ecc_f = 0.0
    inc_0 = 0.0  # rad, baseline
    inc_f = (20.0 * u.deg).to(u.rad).value  # rad
    argp = 0.0  # rad, the method is efficient for 0 and 180
    f = 2.4e-7  # km / s2

    t_domain, r_vectors, v_vectors = _compute_results_array(a, ecc_0, ecc_f, inc_0, inc_f, argp, f)
    ecc_values = _extract_arrays(t_domain, r_vectors, v_vectors)

    # Please, please, save the data
    _save_data(t_domain, ecc_values)

    # TODO: Plotting 70k rows is extremely slow, consider subsampling
    figures = _plot_quantities(t_domain, ecc_values)
    return figures, t_domain, ecc_values


if __name__ == '__main__':
    fig, *_ = plot_geo_case(0.4, 0.0)
    plt.show()
