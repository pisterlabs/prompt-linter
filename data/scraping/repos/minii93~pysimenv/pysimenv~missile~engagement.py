import numpy as np
import matplotlib.pyplot as plt
from pysimenv.core.base import SimObject
from pysimenv.missile.model import PlanarMissile, PlanarVehicle
from pysimenv.missile.guidance import Guidance2dim
from pysimenv.missile.util import RelKin2dim, CloseDistCond, closest_instant, lin_interp


class Engagement2dim(SimObject):
    INTERCEPTED = 1
    MISSILE_STOP = 2
    IS_OUT_OF_VIEW = 3

    def __init__(self, missile: PlanarMissile, target: PlanarVehicle, guidance: Guidance2dim, name="model", **kwargs):
        super(Engagement2dim, self).__init__(name=name, **kwargs)
        self.missile = missile
        self.target = target
        self.guidance = guidance
        self.rel_kin = RelKin2dim(missile, target)
        self.close_dist_cond = CloseDistCond(r_threshold=10.0)

        self._add_sim_objs([self.missile, self.target, self.guidance])

    # override
    def _reset(self):
        super(Engagement2dim, self)._reset()
        self.close_dist_cond.reset()

    # implement
    def _forward(self):
        self.rel_kin.forward()
        self.close_dist_cond.forward(r=self.rel_kin.r)
        lam = self.rel_kin.lam
        sigma = self.missile.look_angle(lam)

        a_M_cmd = self.guidance.forward(self.missile, self.target, self.rel_kin)
        self.missile.forward(a_M_cmd=a_M_cmd)
        self.target.forward()

        self._logger.append(
            t=self.time, r=self.rel_kin.r, sigma=sigma, lam=lam, omega=self.rel_kin.omega
        )

    # implement
    def _check_stop_condition(self) -> bool:
        to_stop = False

        missile_stop = self.missile.check_stop_condition()
        if self.intercepted():  # probable interception
            to_stop = True
            self.flag = self.INTERCEPTED

        if missile_stop:  # stop due to the missile
            to_stop = True
            self.flag = self.MISSILE_STOP

        return to_stop

    def intercepted(self) -> bool:
        return self.close_dist_cond.check()

    def get_info(self) -> dict:
        p_M = self.missile.kin.history('p')
        p_T = self.target.kin.history('p')
        ind_c, xi_c = closest_instant(p_M, p_T)

        p_M_c = lin_interp(p_M[ind_c], p_M[ind_c + 1], xi_c)
        p_T_c = lin_interp(p_T[ind_c], p_T[ind_c + 1], xi_c)
        d_miss = np.linalg.norm(p_M_c - p_T_c)

        gamma_M = self.missile.history('gamma')
        gamma_T = self.target.history('gamma')
        gamma_M_c = lin_interp(gamma_M[ind_c], gamma_M[ind_c + 1], xi_c)
        gamma_T_c = lin_interp(gamma_T[ind_c], gamma_T[ind_c + 1], xi_c)
        gamma_imp = gamma_M_c - gamma_T_c

        t = self.missile.history('t')
        t_imp = lin_interp(t[ind_c], t[ind_c + 1], xi_c)
        return {'miss_distance': d_miss, 'impact_angle': gamma_imp, 'impact_time': t_imp}

    def report(self):
        self.missile.report()
        if self.flag == self.INTERCEPTED:
            print("[engagement] The target has been intercepted!")
        else:
            print("[engagement] The target has been missed!")

        info = self.get_info()
        print("[engagement] Miss distance: {:.6f} (m)".format(info['miss_distance']))
        print("[engagement] Impact angle: {:.2f} (deg)".format(np.rad2deg(info['impact_angle'])))
        print("[engagement] Impact time: {:.2f} (s) \n".format(info['impact_time']))

    def plot_path(self, show=False):
        fig_ax = self.missile.plot_path(label='missile')
        self.target.plot_path(fig_ax=fig_ax, label='target', show=show)

    def plot_rel_kin(self, show=False):
        fig_axs = dict()

        t = self.history('t')
        r = self.history('r')
        sigma = self.history('sigma')
        lam = self.history('lam')
        omega = self.history('omega')
        
        fig, ax = plt.subplots(4, 1, figsize=(6, 8))
        ax[0].set_title("Rel. dist")
        ax[0].plot(t[:-1], r[:-1], label="Rel. dist")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("r (m)")
        ax[0].grid()

        ax[1].set_title("Look angle")
        ax[1].plot(t[:-1], np.rad2deg(sigma[:-1]), label="look angle")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("sigma (deg)")
        ax[1].grid()

        ax[2].set_title("LOS angle")
        ax[2].plot(t[:-1], np.rad2deg(lam[:-1]), label="LOS angle")
        ax[2].set_xlabel("Time (s)")
        ax[2].set_ylabel("lambda (deg)")
        ax[2].grid()

        ax[3].set_title("LOS rate")
        ax[3].plot(t[:-1], np.rad2deg(omega[:-1]), label="LOS rate")
        ax[3].set_xlabel("Time (s)")
        ax[3].set_ylabel("omega (deg/s)")
        ax[3].grid()
        fig.tight_layout()
        fig_axs['Rel. Kin.'] = {'fig': fig, 'ax': ax}

        if show:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.01)

        return fig_axs


