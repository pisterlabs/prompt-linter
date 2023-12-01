# -*- coding: utf-8 -*-
# EditTime  : 2021-09-29 20:22
# Author    : Of yue
# File      : multiMissileGuideInstance.py
# Intro     :
from typing import List

import numpy as np
import multiset as glbs
from numpy import sin, cos, tan, arccos, arctan
from entity.missile import Missile
from guidance.MultiGuide import MultiMissileGuidance
from database.Constant import earth as e
from database.Atmosphere import atmosphereISA as ATM
from utils.common import coords_trans as ct
from utils.common import heading_angle, limit_num
from custom.EntryCorridor import cav_corridor
from utils.interpolate import Interp1
from utils.common import inv_alpha
from core.MultiMissileSim import MultiMissileSim

from scipy import integrate, optimize
import copy


class MultiMisGuideInstance(MultiMissileGuidance):
    def __init__(self):
        super(MultiMisGuideInstance, self).__init__()
        # 各导弹起飞前准备时间
        self.start_time: list = []
        self.control_param_list = []  # 要从制导参数直接导入到控制参数的值名称
        self.guide_phase: dict = {
            "descent_phase1": self.descent_phase1,
            "descent_phase2": self.descent_phase2,
            "steady_glide_phase": self.steady_glide_phase,
            "bias_explicit_guidance": self.bias_explicit_guidance
        }

    def init_custom(self, mis: List[Missile], tar: List[Missile] = [], meta={}):
        self.control_param_list = getattr(glbs, 'CONTROL_PARAM_LIST', ["L", "D", "m", "attack_angle", "bank_angle"])

        for index, (m, t) in enumerate(zip(mis, tar)):
            m.guide.update(
                {
                    # 飞行模式，
                    # default: 默认为正常制导
                    # online: 在线弹道仿真
                    "guide_flag": "default",
                    "alpha_max": m.p.alpha_max,
                    "alpha_sg": m.p.max_L2D_alpha,
                    "alpha_2": np.deg2rad(6),
                    "E2": -5.55e7,
                    "Rs": cav_corridor.average_h_e + e.Re,
                    "VTAEM": glbs.StatusParams[index]["MissileEndStatus"]["velocity"],
                    "HTAEM": glbs.StatusParams[index]["MissileEndStatus"]["height"],
                    "STAEM": glbs.StatusParams[index]["MissileEndStatus"]["s"],
                    "ETAEM": e.E(glbs.StatusParams[index]["MissileEndStatus"]["velocity"],
                                 glbs.StatusParams[index]["MissileEndStatus"]["height"]),
                    "psiTAEM": glbs.StatusParams[index]["MissileEndStatus"]["heading_angle"],
                    "TTAEM": glbs.StatusParams[index]["MissileEndStatus"]["t"],
                    "guide_phase": "descent_phase1",
                    "err_tol": getattr(glbs, 'ERR_TOL', 1e-2),
                    "kgamma": getattr(glbs, 'K_GAMMA', 3),  # 下降段攻角反馈系数
                    "k_PN": getattr(glbs, 'K_PN', 3),  # 显式制导段比例导引系数
                    'k_sigma': getattr(glbs, 'K_SIGMA', -50),  # 最大倾侧角中的系数
                    'k_gamma_sgp': getattr(glbs, 'K_GAMMA_SGP', 3),  # 滑翔段TDCT反馈系数
                    'k_sigma_beg': getattr(glbs, 'K_SIGMA_BEG', 0.5 / 1e5),  # 显式制导反馈系数
                }
            )
            m.guide["EBR"] = [(m.E + m.guide["E2"]) / 2,
                              m.guide["E2"],
                              m.guide["ETAEM"] + 2e6]
            m.guide["BR_flag"] = [False, False, False]
            m.guide["update_EBR_flag"] = [False, False, False]
            m.guide["BR_times"] = 0
            m.guide["L2Dbsl_TAEM"] = cav_corridor.L2D_E(m.guide["ETAEM"])

            # 初始倾侧角符号
            great_circle_heading = heading_angle(m.status.longitude, m.status.latitude,
                                                 t.status.longitude, t.status.latitude)
            m.guide["sgn_ini"] = 1 if great_circle_heading[0] > m.status.heading_angle else -1
            # 设定末端航向角
            m.guide["psiTAEM"] = great_circle_heading[1]
            print(f"导弹{m.guide['index']}初始零偏航向角：{np.rad2deg(great_circle_heading[0])}\n"
                  f"预期末端航向角{np.rad2deg(great_circle_heading[1])}")

            # 其它所有参数设置完后，导弹发射时间初始化
        self.before_online_sim(mis, tar, meta)

        for index, (m, t) in enumerate(zip(mis, tar)):
            # 导弹发射时间
            m.guide["launch_time"] = self.start_time[index]

    def before_online_sim(self, mis: List[Missile], tar: List[Missile] = [], meta={}):
        # 离线仿真阶段，设置导弹的参考弹道与发射时间
        print('设置初始发射时间')
        self.start_time = [0]

    def is_launched(self, mis: Missile, tar: Missile = None, meta={}):
        # 判断各导弹是否发射
        if mis.launched:
            return True
        elif meta["t"] >= mis.guide["launch_time"]:
            mis.launched = True
        return mis.launched

    def guide(self, mis: Missile, tar: Missile = None, meta={}):
        # 决定积分精度
        mis.guide["step_len"] = self.step_len_gen(mis)
        # 解析来自导弹的数据
        self.parse_param(mis, tar, meta)
        # 制导
        self.guide_phase[mis.guide["guide_phase"]](mis, tar, meta)
        # 后续处理，将制导数据传入控制方程
        self.guide2equation(mis, tar, meta)

    def step_len_gen(self, mis: Missile):
        # TODO 根据飞行状态判断积分精度
        return 0.1

    def parse_param(self, mis, tar, meta={}):
        S = mis.p.reference_area
        h = mis.status.height
        v = mis.status.velocity
        E = mis.E
        Ma = mis.ma
        t = mis.status.t
        gamma = mis.status.path_angle
        psi = mis.status.heading_angle
        CL_plan, CD_plan = mis.aero.CLCD(Ma, self.attack_angle_plan(mis, E))
        rho = ATM.rho(h)
        q_inf = rho * v ** 2 * S / 2
        CL, CD = mis.control.get('CL', CL_plan), mis.control.get('CD', CD_plan)
        L, D = CL * q_inf, CD * q_inf
        great_circle_heading = heading_angle(mis.status.longitude, mis.status.latitude,
                                             tar.status.longitude, tar.status.latitude)
        ref_psi = great_circle_heading[0]
        delta_psi = mis.status.heading_angle - ref_psi
        if delta_psi > np.pi:
            delta_psi = delta_psi - 2 * np.pi
        elif delta_psi < -np.pi:
            delta_psi = delta_psi + 2 * np.pi
        sgo = self.s_go(mis, tar, meta)
        q = mis.p.k_Q * np.sqrt(rho) * pow(v, 3.15)
        mis.guide.update({
            'm': mis.p.m,
            'S': S,
            'E': E,
            'Ma': Ma,
            'v': v,
            'h': h,
            'CL': CL,
            'CD': CD,
            'q_inf': q_inf,
            'L': L,
            'D': D,
            'great_circle_heading': great_circle_heading,
            'delta_psi': delta_psi,
            's_go': sgo,
            'ref_psi': ref_psi,
            'q': q,
            'psi': psi,
            'gamma': gamma,
            't': t
        })

    def before_phase(self, mis: Missile, tar: Missile = None, meta={}):
        self.update_kh(mis, tar, meta)
        if not mis.guide["guide_phase"] == 'bias_explicit_guidance':
            if mis.guide["E"] >= mis.guide["EBR"][2] + 1e6:
                # 第三次反转前更新升阻比参数
                self.update_L12D(mis, tar, meta)
            else:
                mis.guide["L12D_param"][1] = mis.guide["L12D_const"]
        self.update_path_angle_sgp(mis, tar, meta)
        if mis.guide["guide_phase"] == "steady_glide_phase":
            self.update_EBR(mis, tar, meta)
        self.BR(mis, tar, meta)

    def descent_phase1(self, mis: Missile, tar: Missile = None, meta={}):
        mis.guide["attack_angle"] = mis.guide["alpha_max"]
        mis.guide["bank_angle"] = 0

        if mis.status.t > 10 and mis.guide["dy"][4] > 0:
            print(f'导弹{mis.guide["index"]}下降段第一段结束, t = {meta["t"]}')
            mis.guide["guide_phase"] = "descent_phase2"
            mis.guide["E_steady_glide_start"] = mis.guide["E"]
            self.before_phase(mis, tar, meta)
            mis.guide["dgamma0"] = mis.guide["gamma_sg"] - mis.status.path_angle

    def descent_phase2(self, mis: Missile, tar: Missile = None, meta={}):
        self.before_phase(mis, tar, meta)
        delta_gamma = mis.guide["gamma_sg"] - mis.status.path_angle
        mis.guide["attack_angle"] = delta_gamma / mis.guide["dgamma0"] * mis.guide["alpha_max"] + (
                mis.guide["dgamma0"] - delta_gamma) / mis.guide["dgamma0"] * (
                                            self.attack_angle_plan(mis, mis.guide["E"]) + mis.guide[
                                        "kgamma"] * delta_gamma)
        mis.guide["bank_angle"] = 0

        if np.abs(delta_gamma) < mis.guide["err_tol"]:
            print(f'下降段第二段结束, t = {meta["t"]}')
            mis.guide["guide_phase"] = "steady_glide_phase"

    def steady_glide_phase(self, mis: Missile, tar: Missile = None, meta={}):
        self.before_phase(mis, tar, meta)
        alpha_bsl = self.attack_angle_plan(mis, mis.guide["E"])
        sigma_bsl = self.sigma_bsl(mis, tar, meta)
        gamma_sg = mis.guide["gamma_sg"]
        sigma_max = self.sigma_max(mis, tar, meta)
        alpha_cmd, sigma_cmd = self.TDCT(mis, alpha_bsl, sigma_bsl, gamma_sg - mis.status.path_angle)
        sigma_cmd = limit_num(sigma_cmd, abs_limit=sigma_max)
        mis.guide["attack_angle"] = alpha_cmd
        mis.guide["bank_angle"] = sigma_cmd

        if mis.guide["E"] < mis.guide["EBR"][2]:
            mis.guide["sgo_beg"] = mis.guide["s_go"]
            mis.guide["guide_phase"] = 'bias_explicit_guidance'
            print(f'平稳滑翔段结束, t = {meta["t"]}')

    def bias_explicit_guidance(self, mis: Missile, tar: Missile = None, meta={}):
        """

        :param mis:
        :param tar:
        :param meta:
        :return:
        """
        alpha, sigma = self.sigma_alpha_beg(mis, tar, meta)
        sigma_max = self.sigma_max(mis, tar, meta)
        sigma = limit_num(sigma, abs_limit=sigma_max)
        alpha = limit_num(alpha, abs_limit=mis.guide["alpha_max"])

        mis.guide["attack_angle"] = alpha
        mis.guide["bank_angle"] = sigma

    def update_kh(self, mis: Missile, tar: Missile, meta={}):
        Rs = mis.guide["Rs"]
        m_phi = mis.status.latitude
        t_phi = tar.status.latitude
        E = mis.guide["E"]
        ET = mis.guide["ETAEM"]
        V = mis.guide["v"]
        VT = mis.guide["VTAEM"]
        H = mis.guide["h"]
        HT = mis.guide["HTAEM"]
        L2DT = mis.guide["L2Dbsl_TAEM"]
        ha = mis.guide["great_circle_heading"]
        w = e.omega_e
        L2D = mis.guide["CL"] / mis.guide["CD"] * cos(mis.control["bank_angle"])
        hz1 = -2 * Rs * w * V * cos(m_phi) * sin(ha[0]) - \
              Rs * w ** 2 * (e.Re + H) * cos(m_phi) * \
              (cos(m_phi) - L2D * sin(m_phi) * cos(ha[0]))
        hz1_t = -2 * Rs * w * VT * cos(t_phi) * sin(ha[1]) - \
                Rs * w ** 2 * (e.Re + HT) * cos(t_phi) * \
                (cos(t_phi) - L2DT * sin(t_phi) * cos(ha[1]))
        kh1 = (hz1_t * E - hz1 * ET) / (E - ET), \
              (hz1 - hz1_t) / (E - ET)

        hz2 = w ** 2 * Rs * (e.Re + H) * L2D * sin(m_phi) * cos(m_phi) * cos(ha[0])
        hz2_t = w ** 2 * Rs * (e.Re + HT) * L2DT * sin(t_phi) * cos(t_phi) * cos(ha[1])
        kh2 = (hz2_t * E - hz2 * ET) / (E - ET), \
              (hz2 - hz2_t) / (E - ET)

        kh3 = -2 * w * Rs * (VT * E - ET * V) * (sin(t_phi) * E - sin(m_phi) * ET) / (E - ET) ** 2, \
              -2 * w * Rs * (VT * sin(m_phi) + V * sin(t_phi)) * (E + ET) / (E - ET) ** 2 \
              + 4 * w * Rs * (VT * sin(t_phi) * E + V * sin(m_phi) * ET) / (E - ET) ** 2, \
              -2 * w * Rs * (V - VT) * (sin(m_phi) - sin(t_phi)) / (E - ET) ** 2

        hz4 = -2 * Rs * w * V * cos(m_phi) * sin(ha[0]) - \
              Rs * w ** 2 * (e.Re + H) * cos(m_phi) ** 2
        hz4_t = -2 * Rs * w * VT * cos(t_phi) * sin(ha[1]) - \
                Rs * w ** 2 * (e.Re + HT) * cos(t_phi) ** 2
        kh4 = (hz4_t * E - hz4 * ET) / (E - ET), \
              (hz4 - hz4_t) / (E - ET)

        mis.guide["kh"] = [
            kh1, kh2, kh3, kh4
        ]

    def update_path_angle_sgp(self, mis: Missile, tar: Missile, meta={}):
        rho = ATM.rho(mis.guide["h"])
        drho_dh = -0.00015 * rho
        v = mis.guide["v"]
        S = mis.guide["S"]
        m = mis.guide["m"]
        Rh = mis.guide["h"] + e.Re
        sigma_bsl = self.sigma_bsl(mis, tar, meta)
        alpha_bsl = self.alpha_bsl(mis, tar, meta)

        CL_bsl, CD_bsl = mis.aero.CLCD(mis.ma, alpha_bsl)
        D_bsl = CD_bsl * mis.guide["q_inf"]
        dCL_dE = cav_corridor.interp_dCL_dE(mis.guide["E"])
        d1 = rho * v ** 2 * S * cos(sigma_bsl) / 2 / m * dCL_dE + 2 / Rh + CL_bsl * rho * S * cos(sigma_bsl) / m
        d2 = -CL_bsl * v ** 2 * S * cos(sigma_bsl) * drho_dh / 2 / e.g0 / m + 2 / Rh + CL_bsl * rho * S * cos(
            sigma_bsl) / m + v ** 2 / Rh ** 2 / e.g0
        mis.guide["gamma_sg"] = -D_bsl / m / e.g0 * d1 / d2

    def alpha_bsl(self, mis: Missile, tar: Missile = None, meta={}):
        return self.attack_angle_plan(mis, mis.guide['E'])

    def sigma_bsl(self, mis: Missile, tar: Missile = None, meta={}):
        if mis.guide["guide_phase"] == 'descent_phase1' or mis.guide["guide_phase"] == 'descent_phase2':
            return 0
        if mis.guide["guide_phase"] == 'steady_glide_phase':
            return self.sigma_bsl_sgp(mis, tar, meta)
        if mis.guide["guide_phase"] == 'bias_explicit_guidance':
            return self.sigma_bsl_beg(mis, tar, meta)

    def sigma_bsl_sgp(self, mis: Missile, tar: Missile = None, meta={}):
        L1_L = self.L12D(mis) / (mis.guide["CL"] / mis.guide["CD"])
        # assert 1 >= L1_L >= -1, '超出射程范围'
        if 1 < L1_L or L1_L < -1:
            return 0
        res = mis.guide["sgn_ini"] * arccos(L1_L)
        return mis.guide["sgn_ini"] * (-1) ** mis.guide["BR_times"] * res

    def sigma_bsl_beg(self, mis: Missile, tar: Missile = None, meta={}):
        # TODO 显式制导倾侧角
        return 0

    def alpha_beg(self, mis: Missile, tar: Missile = None, meta={}):
        # TODO 显式制导攻角
        return 0

    def sigma_alpha_beg(self, mis: Missile, tar: Missile = None, meta={}):
        """
        在线仿真时/显式制导 倾侧角与攻角参数
        :param mis:
        :param tar:
        :param meta:
        :return:
        """
        V = mis.guide["v"]
        gamma = mis.guide["gamma"]
        h = mis.guide["h"]
        phi = mis.status.latitude
        psi = mis.guide["psi"]
        psi_end = mis.guide["psiTAEM"]
        delta_psi_end = 0 if not psi_end else psi_end - psi
        delta_psi = mis.guide["delta_psi"]
        delta_h = mis.guide["HTAEM"] - h
        s_go = mis.guide["s_go"]
        s_go2 = s_go - mis.guide["STAEM"]
        s_LOS = np.linalg.norm([s_go2, delta_h])
        gamma_LOS = np.arctan2(delta_h, s_go2)

        a_tol = e.g0 - V ** 2 / (e.Re + h) - e.omega_e ** 2 * (e.Re + h) * cos(phi) ** 2 - 2 * V * e.omega_e * cos(
            phi) * sin(psi)
        aL1 = -mis.guide["k_PN"] * V ** 2 * sin(gamma - gamma_LOS) / s_LOS + a_tol

        aL2_orig = aL1 * tan(self.sigma_bsl_sgp(mis))
        aL2 = (-12 * V ** 2 * sin(delta_psi) / s_LOS - 6 * V ** 2 * sin(delta_psi_end) / s_LOS) *(mis.guide["sgo_beg"]-s_go)/mis.guide["sgo_beg"] \
              + aL2_orig*s_go2/mis.guide["sgo_beg"]

        mis.guide["a_tol"] = a_tol
        mis.guide["aL2_orig"] = aL2_orig
        mis.guide["aL1"] = aL1
        mis.guide["aL2"] = aL2

        if mis.guide["guide_flag"] == "online":
            # 反解攻角
            CL = mis.guide["m"] * np.linalg.norm([aL1, aL2]) / mis.guide["q_inf"]
            mis.guide["CL_beg"] = CL
            alpha_sim = mis.aero.inverse_alpha(CL, mis.guide["Ma"])
            sigma_sim = arctan(aL2 / aL1)
            return alpha_sim, sigma_sim
        else:
            # sigma = arctan(aL2 / a_tol)
            sigma = arctan(aL2 / aL1)
            # 这里还是采用与参考飞行距离的误差来反馈
            if mis.guide["refs"]:
                k_sgo = 0.2 * s_go / mis.guide["sgo_beg"]
                E_min = mis.guide["refs"].get('ETAEM_ref', -np.inf)
                E_max = mis.guide["refs"].get('E_sim0', np.inf)

                fs_go_ref = mis.guide["refs"].get('f_sgo_ref', None)
                ft_ref = mis.guide["refs"].get('f_t_ref', None)
                fl2d_ref = mis.guide["refs"].get('f_L12D_ref', None)
                E_interp = limit_num(mis.guide["E"], interval_limit=[E_min,E_max],mode='interval')
                sgo_ref = fs_go_ref(E_interp) if fs_go_ref else s_go
                t_ref = ft_ref(E_interp) if ft_ref else mis.guide["t"]
                L12D_ref = fl2d_ref(E_interp) if fl2d_ref else self.L12D(mis, mis.guide["E"])

                s_go_ref2 = sgo_ref - k_sgo * mis.guide["v"] * (mis.guide["t"] - t_ref)

                L12D_beg = L12D_ref + mis.guide["k_sigma_beg"] * (s_go - s_go_ref2)
                # # 调试
                # L12D_beg = L12D_ref
            else:
                L12D_beg = self.L12D(mis, mis.guide["E"])
            mis.guide["ref_sgo"] = sgo_ref
            mis.guide["ref_t"] = t_ref
            mis.guide["ref_L12D"] = L12D_ref
            mis.guide["L12D_beg"] = L12D_beg

            alpha = inv_alpha(mis.guide["Ma"], L12D_beg / cos(sigma), mis.aero)
            return alpha, sigma

    def sigma_max(self, mis: Missile, tar: Missile = None, meta={}):
        if not glbs.CONSIDER_SIGMA_MAX:
            return np.inf
        dHmindE = cav_corridor.dHmindE_E(mis.guide["E"])
        D = mis.control.get('D', mis.guide['D'])
        dHdE = -mis.guide["m"] * sin(mis.status.path_angle) / D
        Hmin = cav_corridor.interp_Hmin_E(mis.guide["E"])
        L1 = mis.guide["m"] * (e.g0 - mis.guide["v"] ** 2 / (e.Re + Hmin)
                               - e.omega_e ** 2 * (e.Re + Hmin) * cos(mis.status.latitude) ** 2
                               - 2 * mis.guide["v"] * e.omega_e * cos(mis.status.latitude)
                               * sin(mis.status.heading_angle))
        Lmax = mis.guide["CL"] * 0.5 * ATM.rho(Hmin) * mis.guide["v"] ** 2 * mis.guide["S"]
        if L1 > Lmax:
            L1 = Lmax
        sigma_max = arccos(L1 / Lmax) + mis.guide["k_sigma"] * (dHmindE - dHdE)
        mis.guide["sigma_max_L1"] = L1
        mis.guide["sigma_max_Lmax"] = Lmax
        mis.guide["sigma_max_dHmindE"] = float(dHmindE)
        mis.guide["sigma_max_dHdE"] = dHdE
        mis.guide["sigma_max"] = sigma_max
        return sigma_max

    def TDCT(self, mis: Missile, alpha_bsl, sigma_bsl, delta_gamma):
        """
        弹道阻尼抑制振荡
        :param alpha_bsl: 攻角预指令
        :param sigma_bsl: 倾侧角预指令
        :param delta_gamma: 平稳滑翔弹道倾角 - 当前弹道倾角
        :return: 攻角指令，倾侧角指令
        """
        alpha_cmd = alpha_bsl + cos(sigma_bsl) * mis.guide["k_gamma_sgp"] * delta_gamma
        sigma_cmd = sigma_bsl - sin(sigma_bsl) * mis.guide["k_gamma_sgp"] * delta_gamma / mis.guide["alpha_sg"]
        return alpha_cmd, sigma_cmd

    def update_EBR(self, mis: Missile, tar: Missile = None, meta={}):
        if not mis.guide["update_EBR_flag"][0]:
            mis.guide["update_EBR_flag"][0] = True
            self.update_EBR1(mis, tar, meta)
        elif not mis.guide["update_EBR_flag"][1] and mis.guide["E"] < mis.guide["EBR"][0]:
            mis.guide["update_EBR_flag"][1] = True
            self.update_EBR2(mis, tar, meta)
        elif not mis.guide["update_EBR_flag"][2] and mis.guide["E"] < mis.guide["EBR"][1]:
            mis.guide["update_EBR_flag"][2] = True
            self.update_EBR3(mis, tar, meta)

    def BR(self, mis: Missile, tar: Missile = None, meta={}):
        if mis.guide["E"] < mis.guide["EBR"][0] and not mis.guide["BR_flag"][0]:
            mis.guide["BR_flag"][0] = True
            mis.guide["BR_times"] = 1
        elif mis.guide["E"] < mis.guide["EBR"][1] and not mis.guide["BR_flag"][1]:
            mis.guide["BR_flag"][1] = True
            mis.guide["BR_times"] = 2

        elif mis.guide["E"] < mis.guide["EBR"][2] and not mis.guide["BR_flag"][2]:

            mis.guide["BR_flag"][2] = True
            mis.guide["BR_times"] = 3

    def update_EBR1(self, mis: Missile, tar: Missile = None, meta={}):
        """
        更新EBR1,EBR2
        :param mis:
        :param tar:
        :param meta:
        :return:
        """
        print("开始更新第一次、第二次反转点")
        res = optimize.root(
            lambda x: self.update_EBR1_fun(mis, x),
            [mis.guide["EBR"][0], mis.guide["EBR"][1]],
            jac=lambda x: self.update_EBR1_jac(mis, x),
            method='lm',
            tol=1e-2
        )
        print(res.message)

    def update_EBR2(self, mis: Missile, tar: Missile = None, meta={}):
        """
        更新EBR2,EBR3
        :param mis:
        :param tar:
        :param meta:
        :return:
        """
        print("开始更新第二次、第三次反转点")
        res = optimize.root(
            lambda x: self.update_EBR2_fun(mis, x),
            [mis.guide["EBR"][1], mis.guide["EBR"][2]],
            jac=lambda x: self.update_EBR2_jac(mis, x),
            method='lm',
            tol=1e-2
        )
        print(res.message)

    def update_EBR3(self, mis: Missile, tar: Missile = None, meta={}):
        """
        更新最后一个反转点及参考升阻比
        :param mis:
        :param tar:
        :param meta:
        :return:
        """
        print(f"开始更新第三次反转点、参考升阻比, E = {mis.guide['E']}")
        # # 方法一
        # VT = mis.guide["VTAEM"]
        # TT = mis.guide["TTAEM"]
        # mis.guide["L12D_const"] = mis.guide["L12D_param"][1]
        # iter_obj0 = np.array([mis.guide["L12D_const"], mis.guide["EBR"][2]])
        # meta["iter_params"] = iter_obj0
        # res = self.simulation_online(mis, tar, meta)
        # vf = res["V"]
        # tf = res["t"]
        # G0 = np.array([vf - VT, 0 if not TT else (tf - TT)])
        # iter_obj1 = np.array([mis.guide["L12D_const"], mis.guide["EBR"][2] - 1e3 * (vf - VT)])
        #
        # for _ in range(glbs.MAX_EBR2_ITER):
        #     meta["iter_params"] = iter_obj1
        #     res = self.simulation_online(mis, tar, meta)
        #     vf = res["V"]
        #     tf = res["t"]
        #     G1 = np.array([vf - VT, 0 if not TT else (tf - TT)])
        #     p = iter_obj1 - iter_obj0
        #     q = np.array([-p[1], p[0]])
        #     q = 0.1 * q / np.linalg.norm(q)
        #     iter_obj2 = iter_obj1 + q
        #     meta["iter_params"] = iter_obj2
        #     res = self.simulation_online(mis, tar, meta)
        #     vf = res["V"]
        #     tf = res["t"]
        #     G2 = np.array([vf - VT, 0 if not TT else (tf - TT)])
        #     theta_p = np.arctan2(p[1], p[0])
        #     theta_q = np.arctan2(q[1], q[0])
        #     inv_temp = np.array([(G1 - G0) / np.linalg.norm(p), (G2 - G1) * 10]).T
        #     inv_jac = np.array([[cos(theta_p), cos(theta_q)],
        #                         [sin(theta_p), sin(theta_q)]]).dot(np.linalg.inv(inv_temp))
        #     iter_obj0 = iter_obj1
        #     temp1 = inv_jac.dot(G1.reshape(2, 1))
        #     iter_obj1 = iter_obj1 - temp1.reshape(2)*0.1
        #     if abs(G1[0]) < 10 and abs(G1[1]) < 10:
        #         break

        # 第二种方法
        VT = mis.guide["VTAEM"]
        TT = mis.guide["TTAEM"]
        mis.guide["L12D_const"] = mis.guide["L12D_param"][1]
        iter_obj0 = np.array([mis.guide["L12D_const"], mis.guide["EBR"][2]])
        meta["iter_params"] = iter_obj0
        res = self.simulation_online(mis, tar, meta)
        vf = res["V"]
        tf = res["t"]
        G0 = np.array([vf - VT, 0 if not TT else (tf - TT)])
        delta_EBR = -1e3 * (vf - VT)
        iter_obj1 = np.array([mis.guide["L12D_const"], mis.guide["EBR"][2] +delta_EBR])

        meta["iter_params"] = iter_obj1
        res = self.simulation_online(mis, tar, meta)
        vf = res["V"]
        tf = res["t"]
        G1 = np.array([vf - VT, 0 if not TT else (tf - TT)])
        iter_obj2 = iter_obj0 + np.array([0.1, 0])
        meta["iter_params"] = iter_obj2
        res = self.simulation_online(mis, tar, meta)
        vf = res["V"]
        tf = res["t"]
        G2 = np.array([vf - VT, 0 if not TT else (tf - TT)])
        p = (G1-G0)/(delta_EBR/1E6)
        q = (G2-G0)*10
        B = np.array([-p[0]*G0[0] - p[1]*G0[1],
                      -q[0]*G0[0] - q[1]*G0[1]])
        temp2 = p[0]*q[0]+p[1]*q[1]
        A = np.array([[1E4+p[0]**2+p[1]**2, temp2],
                      [temp2, 1E4+q[0]**2]+q[1]**2])
        var = np.linalg.solve(A, B)
        var = var*np.array([1,1E6]) + iter_obj0

        meta["iter_params"] = var
        res = self.simulation_online(mis, tar, meta)
        temp_result = res["total_data"]
        mis.guide["L12D_const"] = var[0]
        mis.guide["EBR"][2] = var[1]
        mis.guide["refs"] = {
            "E_sim0":res["E_sim0"],
            "ETAEM_ref": res["E_sim1"],
            "f_sgo_ref": res["interp_sgo_ref"],
            "f_L12D_ref": res["interp_L12D_ref"],
            "f_t_ref": res["interp_t_ref"]
        }


    def update_EBR1_fun(self, mis: Missile, x):
        """
        用于牛顿迭代的方程组
        """
        coef = [1, 1e3]
        mis.guide["EBR"][0] = x[0]
        mis.guide["EBR"][1] = x[1]
        E = mis.guide["E"]
        ETAEM = mis.guide["ETAEM"]
        xc, dpsi = self.xC_dpsi(mis, ETAEM, E)
        return [coef[0] * xc, coef[1] * (dpsi - 0)]

    def update_EBR1_jac(self, mis: Missile, x):
        """
        牛顿迭代雅可比矩阵
        :param mis:
        :param x:
        :return:
        """
        coef = [1, 1e3]
        pxc1, pdpsi1 = self.dxc_dpsi1(mis, mis.guide["E"])
        pxc2, pdpsi2 = self.dxc_dpsi2(mis, mis.guide["E"])
        return np.array([[pxc1, pxc2], [coef[1] * pdpsi1, coef[1] * pdpsi2]])

    def update_EBR2_fun(self, mis: Missile, x):
        """
        用于牛顿迭代的方程组
        """
        coef = [1, 1e3]
        mis.guide["EBR"][1] = x[0]
        mis.guide["EBR"][2] = x[1]
        E = mis.guide["E"]
        ETAEM = mis.guide["ETAEM"]
        xc, dpsi = self.xC_dpsi(mis, ETAEM, E)
        return [coef[0] * xc, coef[1] * (dpsi - 0)]

    def update_EBR2_jac(self, mis: Missile, x):
        """
        牛顿迭代雅可比矩阵
        :param mis:
        :param x:
        :return:
        """
        coef = [1, 1e3]
        pxc2, pdpsi2 = self.dxc_dpsi2(mis, mis.guide["E"])
        pxc3, pdpsi3 = self.dxc_dpsi3(mis, mis.guide["E"])
        return np.array([[pxc2, pxc3], [coef[1] * pdpsi2, coef[1] * pdpsi3]])

    def update_L12D(self, mis: Missile, tar: Missile, meta={}):
        # TODO 除去重复代码
        E0 = mis.guide["E_steady_glide_start"]
        E1 = mis.guide["E2"]
        E2 = mis.guide["ETAEM"]
        L2DTAEM = mis.guide["L2Dbsl_TAEM"]

        c1 = [0, 1 / (E0 - E1), -E1 / (E0 - E1)]
        c2 = [0, 1 / (E1 - E0), E0 / (E0 - E1)]
        c3 = [1 / (E1 - E2) ** 2, -2E1 / (E1 - E2) ** 2, E1 ** 2 / (E1 - E2) ** 2]
        c4 = [0, 0, 1]
        c4_c3 = [-1 / (E1 - E2) ** 2, 2E1 / (E1 - E2) ** 2, 1 - E1 ** 2 / (E1 - E2) ** 2]

        # 这里终端能量取ETAEM
        # TODO 补齐时间误差引起的升阻比
        if mis.guide["E"] > E1:
            L12D11 = ((mis.guide["s_go"] - mis.guide["STAEM"])/cos(mis.guide["delta_psi"]) - L2DTAEM * self.x_D(mis, E2, E1, c=c3)) \
                     / (self.x_D(mis, E1, mis.guide["E"], c=[0, 0, 1]) + self.x_D(mis, E2, E1, c=c4_c3))
        else:
            L12D11 = ((mis.guide["s_go"] - mis.guide["STAEM"])/cos(mis.guide["delta_psi"]) - L2DTAEM * self.x_D(mis, E2, mis.guide["E"], c=c3)) \
                     / self.x_D(mis, E2, mis.guide["E"], c=c4_c3)

        L12D21 = L12D11
        L12D12 = 0
        L12D22 = 0

        L2DE = L12D11 + L12D12
        L2D_alpha = L12D21 + L12D22
        mis.guide["L12D_param"] = [L2DE, L2D_alpha]

    def L12D(self, mis: Missile, E=None):
        if not E:
            E = mis.guide["E"]
        E0 = mis.guide["E_steady_glide_start"]
        E1 = mis.guide["E2"]
        E2 = mis.guide["ETAEM"]
        params = mis.guide["L12D_param"]
        L2DTAEM = mis.guide["L2Dbsl_TAEM"]
        L2DE = params[0]
        L2D_alpha = params[1]
        if E >= E1:
            return (E - E1) / (E0 - E1) * L2DE + (E0 - E) / (E0 - E1) * L2D_alpha
        else:
            return ((E1 - E) / (E1 - E2)) ** 2 * (L2DTAEM - L2D_alpha) + L2D_alpha

    def L22D(self, mis: Missile, E=None):
        if not E:
            E = mis.guide["E"]
        return abs(np.sqrt(cav_corridor.L2D_E(E) ** 2 - self.L12D(mis, E) ** 2))

    def x_D(self, mis, E_end, E_start, c=None):
        """
        射程解析解
        :param mis:
        :param E_end:
        :param E_start:
        :param c: 升阻比形式决定的系数(a2,a1,a0)
        :return:
        """
        if E_start < E_end:
            return 0

        if c is None:
            # 如果不输入c的值，则认为是输出总射程，要乘上升阻比
            E0 = mis.guide["E_steady_glide_start"]
            E1 = mis.guide["E2"]
            E2 = mis.guide["ETAEM"]
            params = mis.guide["L12D_param"]
            L2DTAEM = mis.guide["L2Dbsl_TAEM"]
            L2DE = params[0]
            L2D_alpha = params[1]
            c1 = [0, 1 / (E0 - E1), -E1 / (E0 - E1)]
            c2 = [0, 1 / (E1 - E0), E0 / (E0 - E1)]
            c3 = [1 / (E1 - E2) ** 2, -2E1 / (E1 - E2) ** 2, E1 ** 2 / (E1 - E2) ** 2]
            c4 = [0, 0, 1]
            if E_end >= E1:
                return L2DE * self.x_D(mis, E_end, E_start, c=c1) + L2D_alpha * self.x_D(mis, E_end, E_start, c=c2)
            elif E_start <= E1:
                return (L2DTAEM - L2D_alpha) * self.x_D(mis, E_end, E_start, c=c3) + L2D_alpha * self.x_D(mis, E_end,
                                                                                                          E_start, c=c4)
            else:
                return L2DE * self.x_D(mis, E1, E_start, c=c1) + L2D_alpha * self.x_D(mis, E1, E_start, c=c2) \
                       + (L2DTAEM - L2D_alpha) * self.x_D(mis, E_end, E1, c=c3) \
                       + L2D_alpha * self.x_D(mis, E_end, E1, c=c4)
        # 如果输入c，则要计算各子段的射程（不乘升阻比）
        a2, a1, a0 = c[0], c[1], c[2]
        kh1 = mis.guide["kh"][0]
        kh4 = mis.guide["kh"][3]

        Rs = mis.guide["Rs"]
        temp1 = a1 - e.mu / 2 / Rs * a2
        temp2 = (e.mu / 2 / Rs) ** 2 * a2 - e.mu / 2 / Rs * a1 + a0
        temp3 = np.log((2 * E_end + e.mu / Rs) / (2 * E_start + e.mu / Rs))
        temp4 = 1 / (2 * E_end + e.mu / Rs) - 1 / (2 * E_start + e.mu / Rs)
        fx1 = e.Re * a2 / 4 * (E_end ** 2 - E_start ** 2) + e.Re / 2 * temp1 * (
                E_end - E_start) + e.Re / 2 * temp2 * temp3
        fx2 = e.Re / 2 * (a2 * (E_end - E_start) / 2 + (a1 - e.mu / Rs * a2) / 2 * temp3
                          - temp2 * temp4)
        fx3 = e.Re / 8 * a2 * temp3 - e.Re / 4 * (a1 - e.mu / Rs * a2) * temp4 - e.Re / 4 * temp2 * (
                1 / (2 * E_end + e.mu / Rs) ** 2 - 1 / (2 * E_start + e.mu / Rs) ** 2)
        res = (1 + kh1[1] / 2 + kh1[1] * kh4[1] / 4) * fx1 + \
              (kh1[0] * (1 + kh4[1] / 2) - kh1[1] / 2 * (e.mu / Rs - kh4[0]) - e.mu / 2 / Rs * kh1[1] * kh4[1]) * fx2 + \
              (kh1[0] - e.mu / 2 / Rs * kh1[1]) * (kh4[0] - e.mu / 2 / Rs * kh4[1]) * fx3
        return res

    def h_m(self, mis: Missile, E=None):
        if not E:
            E = mis.guide["E"]
        return 2 * E + e.mu / mis.guide["Rs"]

    def xC_dpsi(self, mis, E_end=None, E_start=None):
        if not E_end:
            E_end = mis.guide["ETAEM"]
        if not E_start:
            E_start = mis.guide["E"]

        EBR = mis.guide["EBR"]
        delta_psi = mis.guide["delta_psi"]
        integ_xc = integrate.quad(
            lambda x: sin(self.x_D(mis, E_end, x) / e.Re) * self.f2(mis, x),
            E_start, E_end
        )
        integ_dpsi = integrate.quad(
            lambda x: cos(self.x_D(mis, E_end, x) / e.Re) * self.f2(mis, x),
            E_start, E_end
        )
        # 判断是第几次反转后
        if E_start > EBR[0]:
            F1, G1 = self.FG(mis, EBR[0], E_start)
            F2, G2 = self.FG(mis, EBR[1], EBR[0])
        elif E_start >= EBR[1]:
            F1, G1 = 0, 0
            F2, G2 = self.FG(mis, EBR[1], E_start)
        else:
            raise "在第二次反转后不采用解析解更新反转点"
        F3, G3 = self.FG(mis, EBR[2], EBR[1])
        F4, G4 = self.FG(mis, E_end, EBR[2])
        xc = e.Re * (delta_psi * sin(self.x_D(mis, E_end, E_start) / e.Re) - integ_xc[0]
                     - mis.guide["sgn_ini"] * (F1 - F2 + F3 - F4))
        dpsi = delta_psi * cos(self.x_D(mis, E_end, E_start) / e.Re) - integ_dpsi[0] \
               - mis.guide["sgn_ini"] * (G1 - G2 + G3 - G4)
        return xc, dpsi

    def dxc_dpsi1(self, mis: Missile, E):
        xd = self.x_D(mis, mis.guide["ETAEM"], mis.guide["EBR"][0])
        f4 = self.f4(mis, mis.guide["EBR"][0])
        pxc = -2 * mis.guide["sgn_ini"] * e.Re * sin(xd / e.Re) * f4
        pdpsi = -2 * mis.guide["sgn_ini"] * cos(xd / e.Re) * f4
        return pxc, pdpsi

    def dxc_dpsi2(self, mis: Missile, E):
        xd = self.x_D(mis, mis.guide["ETAEM"], mis.guide["EBR"][1])
        f4 = self.f4(mis, mis.guide["EBR"][1])
        pxc = 2 * mis.guide["sgn_ini"] * e.Re * sin(xd / e.Re) * f4
        pdpsi = 2 * mis.guide["sgn_ini"] * cos(xd / e.Re) * f4
        return pxc, pdpsi

    def dxc_dpsi3(self, mis: Missile, E):
        xd = self.x_D(mis, mis.guide["ETAEM"], mis.guide["EBR"][2])
        f4 = self.f4(mis, mis.guide["EBR"][2])
        pxc = -2 * mis.guide["sgn_ini"] * e.Re * sin(xd / e.Re) * f4
        pdpsi = -2 * mis.guide["sgn_ini"] * cos(xd / e.Re) * f4
        return pxc, pdpsi

    def f2(self, mis: Missile, E):
        kh3 = mis.guide["kh"][2]
        kh4 = mis.guide["kh"][3]
        Rs = mis.guide["Rs"]
        L12D = self.L12D(mis, E)
        return -(L12D * (kh3[0] + kh3[1] * E + kh3[2] * E ** 2)) / (
                (2 * E + 2 * e.mu / Rs) * (2 * E + e.mu / Rs)
        ) * (1 + (kh4[0] + kh4[1] * E) / (2 * E + e.mu / Rs))

    def f4(self, mis: Missile, E):
        if not E:
            E = mis.guide["E"]
        Rs = mis.guide["Rs"]
        L22D = self.L22D(mis, E)
        kh2 = mis.guide["kh"][1]
        kh4 = mis.guide["kh"][3]
        h2 = kh2[0] + kh2[1] * E
        h4 = kh4[0] + kh4[1] * E
        return -L22D / (2 * E + 2 * e.mu / Rs) * (
                1 + h2 / (2 * E + e.mu / Rs) +
                (h2 * h4) / (2 * E + e.mu / Rs) ** 2
        )

    def FG(self, mis: Missile, E_end, E_start, E_xD=None):
        if not E_xD:
            E_xD = mis.guide["ETAEM"]
        F = integrate.quad(lambda x: sin(self.x_D(mis, E_xD, x) / e.Re) * self.f4(mis, x),
                           E_start, E_end, epsabs=1e-7)
        G = integrate.quad(lambda x: cos(self.x_D(mis, E_xD, x) / e.Re) * self.f4(mis, x),
                           E_start, E_end, epsabs=1e-7)
        return F[0], G[0]

    def simulation_online(self, mis: Missile, tar: Missile = None, meta={}):
        sim: MultiMissileSim = copy.deepcopy(meta["simulation"])
        sim.is_main = False
        index = mis.guide["index"]
        for m in sim.mis:
            m.guide["guide_flag"] = "online"
            if not m.guide["index"] == index:
                m.guide["end_guide"] = True
            else:
                m.guide["L12D_const"] = meta["iter_params"][0]
                m.guide["EBR"][2] = meta["iter_params"][1]
        sim.simulation()
        result = sim.db[index].data
        misT = sim.mis[index]
        s_go = result['s_go']
        E_node = result['E']
        interp_sgo_ref = Interp1(E_node, s_go).pre
        L12D = result["L12D"]
        interp_L12D_ref = Interp1(E_node, L12D).pre
        t_list = result["t"]
        interp_t_ref = Interp1(E_node, t_list).pre
        V = misT.guide["v"]
        t = misT.status.t
        return {
            "E_sim0":mis.guide["E"],
            "E_sim1":misT.guide["E"],
            "V": V,
            "t": t,
            "interp_sgo_ref": interp_sgo_ref,
            "interp_L12D_ref": interp_L12D_ref,
            "interp_t_ref": interp_t_ref,
            "total_data": result
        }

    def guide2equation(self, mis: Missile, tar: Missile = None, meta={}):
        for item in self.control_param_list:
            mis.control[item] = mis.guide[item]
        mis.control['CL'], mis.control['CD'] = mis.aero.CLCD(mis.ma, mis.guide["attack_angle"])
        mis.control['L'] = mis.control['CL'] * mis.guide['q_inf']
        mis.control['D'] = mis.control['CD'] * mis.guide['q_inf']
        mis.guide["L12D"] = mis.control['CL'] / mis.control['CD'] * cos(mis.guide["bank_angle"])
        mis.guide["L12D_E"] = mis.guide.get('L12D_param', [None, None])[0]
        mis.guide["L12D_alpha"] = mis.guide.get('L12D_param', [None, None])[1]

    def attack_angle_plan(self, mis, E=None):
        if not E:
            E = mis.guide["E"]
        a1 = mis.guide["alpha_sg"]
        a2 = mis.guide["alpha_2"]
        E2 = mis.guide["E2"]
        E_TAEM = mis.guide["ETAEM"]
        return [a1, ((E2 - E) / (E2 - E_TAEM)) ** 2 * (a2 - a1) + a1][E < E2]

    def s_go(self, mis: Missile, tar: Missile = None, meta={}):
        x_mis = ct.coordinate_transformation(0, lamb=mis.status.longitude, phi=mis.status.latitude)
        x_tar = ct.coordinate_transformation(0, lamb=tar.status.longitude, phi=tar.status.latitude)
        return e.Re * arccos(x_mis.dot(x_tar))

    def mis_end_guide(self, mis: Missile, tar: Missile, meta={}):
        if mis.guide.get('s_go', np.inf) < mis.guide["STAEM"]:
            mis.guide["end_guide"] = True
        if mis.guide.get('E', np.inf) < (mis.guide["ETAEM"] - 5E5) and mis.guide["guide_flag"] == 'online':
            mis.guide["end_guide"] = True
        # if mis.status.t > 1000:
        #     mis.guide["end_guide"] = True
