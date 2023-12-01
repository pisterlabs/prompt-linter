import numpy as np
import settings as glbs
from numpy import sin, cos, tan, arccos, arctan
from entity.missile import Missile
from guidance.guide import Guidance
from database.Constant import earth as e
from database.Atmosphere import atmosphereISA as ATM
from utils.common import coords_trans as ct
from utils.common import heading_angle, limit_num
from custom.EntryCorridor import cav_corridor
from utils.interpolate import Interp1
from simulation import TrajectorySimulation

from scipy import integrate, optimize
import copy


# 计算制导参数


class CustomGuidance(Guidance):
    def __init__(self):
        super(CustomGuidance, self).__init__()
        # 制导模式
        self.guide_mode = 0
        self.guide_phase: dict = {
            "descent_phase1": self.descent_phase1,
            "descent_phase2": self.descent_phase2,
            "steady_glide_phase": self.steady_glide_phase,
            "altitude_adjustment_phase": self.altitude_adjustment_phase
        }
        # 参数初始化
        self.alpha_max = 0  # 导弹最大攻角
        self.alpha_sgp = 0  # 平稳滑翔攻角
        self.alpha_2 = np.deg2rad(6)
        self.V_TAEM = 0  # 终端速度
        self.E_TAEM = 0  # 终端能量
        self.H_TAEM = 0  # 终端高度
        self.E_alpha = -5.55e7
        self.S_TAEM = 0  # 终端距离
        self.EBR1 = 0  # 首次倾侧角翻转能量
        self.EBR2 = 0  # 二次倾侧角翻转能量
        self.sgn = 1  # 初始倾侧角方向
        self.kgamma = 0  # 下降段攻角反馈系数
        self.err_tol = 0  # 进入平稳滑翔段时弹道倾角与平稳滑翔弹道倾角的容忍误差
        self.EBR1_update_flag = [False, False, False]  # 三次更新EBR1
        self.EBR2_update_flag = [False, False]  # 两次更新EBR2
        self.bank_reverse_flag = [False, False]  # 两次倾侧角反转
        self.glide_E0 = 0  # 初次进入滑翔段的能量
        self.bank_reverse_time = 0  # 倾侧角反转次数，实际上用不到，因为只反转两次，可以在sigma_bsl中用两种情况表示
        self.k_gamma_sgp = 0  # 滑翔段TDCT反馈系数
        self.k_gamma_aap = 0  # 姿态调整段倾侧角反馈系数
        self.control_param_list = []  # 要从制导参数直接导入到控制参数的值名称
        self.k_sigma = 0  # 最大倾侧角中的系数
        self.k_alpha = 0  # AAP中攻角对于飞行距离偏差的反馈系数
        self.allow_update_param = True  # 是否允许更新EBR2与alpha2，如果是在线制导，则设为False
        self.accurate_mode = 0  # 仿真精度

    def init(self, missile: Missile, target: Missile = None, meta={}):
        print('正在初始化参数')
        self.alpha_max = missile.p.alpha_max
        self.alpha_sgp = missile.p.max_L2D_alpha
        self.V_TAEM = glbs.MissileEndStatus["velocity"]
        self.H_TAEM = glbs.MissileEndStatus["height"]
        self.E_TAEM = e.E(self.V_TAEM, self.H_TAEM)
        self.S_TAEM = glbs.MissileEndStatus["s"]
        self.guide_mode = "descent_phase1"
        self.EBR1 = (missile.E + self.E_alpha) / 2
        self.EBR2 = self.E_alpha
        self.err_tol = getattr(glbs, 'ERR_TOL', 1e-2)
        self.kgamma = getattr(glbs, 'K_GAMMA', 3)
        self.k_gamma_sgp = getattr(glbs, 'K_GAMMA_SGP', 3)
        self.k_gamma_aap = getattr(glbs, 'K_GAMMA_AAP', 3)
        self.k_sigma = getattr(glbs, 'K_SIGMA', -50)
        self.k_alpha = getattr(glbs, 'K_ALPHA', 5 * np.pi / 1.8e7)
        self.control_param_list = getattr(glbs, 'CONTROL_PARAM_LIST', ["L", "D", "m", "attack_angle", "bank_angle"])
        # 初始方向角偏大则倾侧角为负
        great_circle_heading = heading_angle(missile.status.longitude, missile.status.latitude,
                                             target.status.longitude, target.status.latitude)
        self.sgn = 1 if great_circle_heading[0] > missile.status.heading_angle else -1

        # 只需要计算一次的制导参数初始化
        missile.guide["L2Dbsl_TAEM"] = cav_corridor.L2D_E(self.E_TAEM)

    def guide(self, mis: Missile, tar: Missile = None, meta={}):
        self.parse_param(mis, tar, meta)
        self.integral_accurate(mis, tar, meta)
        self.guide_phase[self.guide_mode](mis, tar, meta)
        self.guide2control(mis, tar, meta)

        # 调试时控制程序停止：
        # if self.guide_mode == "steady_glide_phase":
        #     self.end_guide(flag=True)
        # if meta["t"] >= 1000:
        #     self.end_guide(flag=True)

    def parse_param(self, mis, tar, meta={}):
        """
        解析导弹及目标参数
        :param mis: 导弹
        :param tar: 目标
        :return:
        """
        S = mis.p.reference_area
        h = mis.status.height
        v = mis.status.velocity

        psi = mis.status.heading_angle
        E = mis.E
        Ma = mis.ma
        CL_plan, CD_plan = mis.aero.CLCD(Ma, self.attack_angle_plan(E))
        rho = ATM.rho(h)
        q_inf = rho * v ** 2 * S / 2
        # 真实升阻系数按照上一步来取，如果是第一次，则取参考值
        CL, CD = mis.control.get('CL', CL_plan), mis.control.get('CD', CD_plan)
        L, D = CL * q_inf, CD * q_inf
        great_circle_heading = heading_angle(mis.status.longitude, mis.status.latitude,
                                             tar.status.longitude, tar.status.latitude)
        ref_psi = great_circle_heading[0]
        delta_psi = psi - ref_psi
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
        })

    def integral_accurate(self, mis: Missile, tar: Missile = None, meta={}):
        self.accurate_mode = 0
        d_EBR1 = mis.guide["E"] - self.EBR1
        d_EBR2 = mis.guide["E"] - self.EBR2
        d_s_go = mis.guide["s_go"] - self.S_TAEM
        if self.guide_mode == 'descent_phase1' or self.guide_mode == 'descent_phase2':
            self.accurate_mode = 1
        if 0 < d_EBR1 < 1e5:
            self.accurate_mode = 1
        if 0 < d_EBR2 < 1e5:
            if d_EBR2 < 1e3:
                self.accurate_mode = 3
            elif d_EBR2 < 1e4:
                self.accurate_mode = 2
            else:
                self.accurate_mode = 1
        if 0 < d_s_go < 1e4:
            if d_s_go < 2e3:
                self.accurate_mode = 3
            elif d_s_go < 5e3:
                self.accurate_mode = 2
            else:
                self.accurate_mode = 1

    def attack_angle_plan(self, E):
        a1 = self.alpha_sgp
        a2 = self.alpha_2
        return [a1, ((self.E_alpha - E) / (self.E_alpha - self.E_TAEM)) ** 2 * (a2 - a1) + a1][E < self.E_alpha]

    def descent_phase1(self, mis: Missile, tar: Missile = None, meta={}):
        """
        mode: descent_phase1
        初始下降段，当\dot{\gamma}<0时，输出最大攻角
        next_mode: 当dot gamma = 0 （第一次>0）时，记录倾角偏差值，进入descent_phase2
        :return: None
        """
        # 导弹第一次制导即启动
        mis.launched = True

        mis.guide["attack_angle"] = self.alpha_max
        mis.guide["bank_angle"] = 0

        if mis.status.t > 10 and mis.guide["dy"][4] > 0:
            print(f'下降段第一段结束, t = {meta["t"]}')
            self.guide_mode = "descent_phase2"
            self.before_phase(mis, tar, meta)
            mis.guide["dgamma0"] = mis.guide["gamma_sg"] - mis.status.path_angle

    def descent_phase2(self, mis: Missile, tar: Missile = None, meta={}):
        """
        下降段的第二段，用来调整攻角，平稳过渡到滑翔段
        :param mis:
        :param tar:
        :param meta:
        next_mode: steady_glide_phase. 当弹道倾角与平稳滑翔弹道倾角小于setting中的err_tol时进入下一阶段
        :return:
        """
        self.before_phase(mis, tar, meta)
        delta_gamma = mis.guide["gamma_sg"] - mis.status.path_angle
        mis.guide["attack_angle"] = delta_gamma / mis.guide["dgamma0"] * self.alpha_max + (
                mis.guide["dgamma0"] - delta_gamma) / mis.guide["dgamma0"] * (
                                            self.attack_angle_plan(mis.guide["E"]) + self.kgamma * delta_gamma)
        mis.guide["bank_angle"] = 0

        if np.abs(delta_gamma) < self.err_tol:
            print(f'下降段第二段结束, t = {meta["t"]}')
            self.guide_mode = "steady_glide_phase"
            self.glide_E0 = mis.guide["E"]

    def steady_glide_phase(self, mis: Missile, tar: Missile = None, meta={}):
        """
        平稳滑翔段
        :param mis:
        :param tar:
        :param meta:
        next_mode: altitude_adjustment_phase. 当第二次反转倾侧角后进入下一段.
        :return:
        """
        self.before_phase(mis, tar, meta)
        alpha_bsl = self.attack_angle_plan(mis.guide["E"])
        sigma_bsl = self.sigma_bsl(mis, tar, meta)
        gamma_sg = mis.guide["gamma_sg"]
        sigma_max = self.sigma_max(mis, tar, meta)
        alpha_cmd, sigma_cmd = self.TDCT(alpha_bsl, sigma_bsl, gamma_sg - mis.status.path_angle)
        sigma_cmd = limit_num(sigma_cmd, abs_limit=sigma_max)
        mis.guide["attack_angle"] = alpha_cmd
        mis.guide["bank_angle"] = sigma_cmd

        if mis.guide["E"] < self.EBR2:
            mis.guide["sgo_EBR2"] = mis.guide["s_go"]
            self.guide_mode = "altitude_adjustment_phase"
            print(f'平稳滑翔段结束, t = {meta["t"]}')

    def altitude_adjustment_phase(self, mis: Missile, tar: Missile = None, meta={}):
        self.before_phase(mis, tar, meta)
        alpha_bsl = self.attack_angle_plan(mis.guide["E"])
        sigma_bsl = self.sigma_bsl(mis, tar, meta)
        f_sgo_ref = mis.guide.get('f_sgo_ref', None)
        sgo_ref = f_sgo_ref(mis.guide["E"]) if f_sgo_ref else mis.guide["s_go"]
        mis.guide["sgo_ref"] = sgo_ref
        delta_gamma = mis.guide["gamma_sg"] - mis.status.path_angle

        alpha_cmd = alpha_bsl + self.k_alpha * (mis.guide["s_go"] - sgo_ref)
        sigma_cmd = sigma_bsl - sin(sigma_bsl) * self.k_gamma_aap * delta_gamma / self.alpha_sgp

        mis.guide["attack_angle"] = alpha_cmd
        mis.guide["bank_angle"] = sigma_cmd

    def before_phase(self, mis: Missile, tar: Missile = None, meta={}):
        self.update_kh(mis, tar, meta)
        if not self.guide_mode == 'altitude_adjustment_phase':
            self.update_L1D(mis, tar, meta)
        self.update_path_angle_sgp(mis, tar, meta)
        if self.guide_mode == 'steady_glide_phase':
            if self.need_update_EBR1(mis, tar, meta):
                self.update_EBR1(mis, tar, meta)
            if self.need_update_EBR2(mis, tar, meta):
                self.update_EBR2_alpha2(mis, tar, meta)
        self.bank_reverse(mis, tar, meta)

    def need_update_EBR1(self, mis: Missile, tar: Missile = None, meta={}):
        if not self.EBR1_update_flag[0]:
            if self.guide_mode == 'steady_glide_phase':
                self.EBR1_update_flag[0] = True
                return True
        elif not self.EBR1_update_flag[1]:
            if mis.guide["E"] < (self.EBR1 + self.glide_E0) / 2:
                self.EBR1_update_flag[1] = True
                return True
        elif not self.EBR1_update_flag[2]:
            dEdt = mis.control["D"] * mis.guide["v"] / mis.guide["m"]
            delta_t = 100
            if mis.guide["E"] < (self.EBR1 + dEdt * delta_t):
                self.EBR1_update_flag[2] = True
                return True
        return False

    def need_update_EBR2(self, mis: Missile, tar: Missile = None, meta={}):
        if not self.allow_update_param:
            return False
        if not self.EBR2_update_flag[0]:
            if self.bank_reverse_flag[0]:
                self.EBR2_update_flag[0] = True
                return True
        elif not self.EBR2_update_flag[1]:
            dEdt = mis.control["D"] * mis.guide["v"] / mis.guide["m"]
            delta_t = 120
            if mis.guide["E"] < (self.EBR2 + dEdt * delta_t):
                self.EBR2_update_flag[1] = True
                return True
        return False

    def update_EBR2_alpha2(self, mis: Missile, tar: Missile = None, meta={}):
        print(f'\n在线更新alpha2: t = {meta["t"]}')
        self.update_alpha2(mis, tar, meta)
        print('在线更新alpha2完毕\n')
        print(f'\n在线更新EBR2: t = {meta["t"]}')
        # 只更新一次EBR2
        # if self.EBR2_update_flag[1] == False:
        #     self.update_EBR2(mis, tar, meta)
        self.update_EBR2(mis, tar, meta)
        print('在线更新EBR2完毕\n')
        # if not self.EBR2_update_flag[1]:
        #     print('\n在线更新EBR2:')
        #     self.update_EBR2(mis, tar, meta)
        #     print('在线更新EBR2完毕\n')
        #     print('\n在线更新alpha2:')
        #     self.update_alpha2(mis, tar, meta)
        #     print('在线更新alpha2完毕\n')
        # else:
        #     print('\n在线更新alpha2:')
        #     self.update_alpha2(mis, tar, meta)
        #     print('在线更新alpha2完毕\n')
        #     print('\n在线更新EBR2:')
        #     self.update_EBR2(mis, tar, meta)
        #     print('在线更新EBR2完毕\n')

    def update_alpha2(self, mis: Missile, tar: Missile = None, meta={}):
        sim = copy.deepcopy(meta["simulation"])
        f_stat = self.simulation_online(sim)
        VT = self.V_TAEM
        m = mis.guide['m']
        alpha_bslf = f_stat['alpha_bsl']
        alpha2f = f_stat['alpha2']
        CLf = f_stat['CL']
        Hf = f_stat['h']
        vf = f_stat['v']
        phif = f_stat['phi']
        psif = f_stat['psi']
        CL_alpha = mis.aero.CL_alpha(vf / ATM.a(Hf))
        q_Sf = f_stat['q_S']
        q_S_T = 0.5 * ATM.rho(self.H_TAEM) * VT ** 2 * mis.p.reference_area
        self.alpha_2 = alpha_bslf + CLf * (q_Sf - q_S_T) / (CL_alpha * q_S_T) \
                       + m * (vf ** 2 - VT ** 2) / (CL_alpha * q_S_T * (e.Re + Hf)) \
                       + 2 * m * (vf - VT) * e.omega_e * cos(phif) * sin(psif) / (CL_alpha * q_S_T)

    def update_EBR2(self, mis: Missile, tar: Missile = None, meta={}):
        sim_old = copy.deepcopy(meta["simulation"])
        sim_result_old = self.simulation_online(sim_old)
        vf_old = sim_result_old['v']
        interp_sgo_ref = sim_result_old['interp_sgo_ref']
        EBR2_old, EBR2_new = self.EBR2, self.EBR2 - 1e3 * (vf_old - self.V_TAEM)
        for _ in range(glbs.MAX_EBR2_ITER):
            sim_new = copy.deepcopy(meta["simulation"])
            sim_new.guide.EBR2 = EBR2_new
            sim_result = self.simulation_online(sim_new)
            vf_new = sim_result['v']
            interp_sgo_ref = sim_result['interp_sgo_ref']
            EBR2_update = EBR2_new - (vf_new - self.V_TAEM) * (EBR2_new - EBR2_old) / (vf_new - vf_old)
            EBR2_old, vf_old = EBR2_new, vf_new
            EBR2_new = EBR2_update
            if np.abs(vf_new - self.V_TAEM) < 1:
                break
        self.EBR2 = EBR2_new
        # 保存参考距离
        # if not self.EBR2_update_flag[1]:
        mis.guide['f_sgo_ref'] = interp_sgo_ref

    def simulation_online(self, sim: TrajectorySimulation):
        # 使用现有状态进行弹道仿真，传回最终状态及参考飞行距离
        sim.guide.allow_update_param = False
        sim.is_online = True
        sim.guide.k_alpha = 0
        sim.simulation()
        result = sim.db.data

        # CL, q_inf, v, h, phi, psi, alpha
        E = sim.mis.E
        alpha_bsl = sim.guide.attack_angle_plan(E)
        alpha2 = sim.guide.alpha_2
        q_S = sim.mis.guide['q_inf']  # 这里统一不除以S
        v = sim.mis.guide["v"]
        h = sim.mis.guide["h"]
        CL = sim.mis.guide["CL"]
        phi = sim.mis.status.latitude
        psi = sim.mis.status.heading_angle
        s_go = result['s_go']
        E_node = result['E']
        interp_sgo_ref = Interp1(E_node, s_go).pre
        res = {
            'alpha_bsl': alpha_bsl,
            'alpha2': alpha2,
            'q_S': q_S,
            'v': v,
            'h': h,
            'CL': CL,
            'phi': phi,
            'psi': psi,
            'interp_sgo_ref': interp_sgo_ref
        }
        return res

    def bank_reverse(self, mis: Missile, tar: Missile = None, meta={}):
        if not self.bank_reverse_flag[0]:
            if mis.guide["E"] < self.EBR1:
                self.bank_reverse_flag[0] = True
                self.bank_reverse_time = self.bank_reverse_time + 1
                print(f'第一次倾侧角反转, t = {meta["t"]}')
        elif not self.bank_reverse_flag[1]:
            if mis.guide["E"] < self.EBR2:
                self.bank_reverse_flag[1] = True
                self.bank_reverse_time = self.bank_reverse_time + 1
                print(f'第二次倾侧角反转, t = {meta["t"]}')

    def TDCT(self, alpha_bsl, sigma_bsl, delta_gamma):
        """
        弹道阻尼抑制振荡
        :param alpha_bsl: 攻角预指令
        :param sigma_bsl: 倾侧角预指令
        :param delta_gamma: 平稳滑翔弹道倾角 - 当前弹道倾角
        :return: 攻角指令，倾侧角指令
        """
        alpha_cmd = alpha_bsl + cos(sigma_bsl) * self.k_gamma_sgp * delta_gamma
        sigma_cmd = sigma_bsl - sin(sigma_bsl) * self.k_gamma_sgp * delta_gamma / self.alpha_sgp
        return alpha_cmd, sigma_cmd

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
        sigma_max = arccos(L1 / Lmax) + self.k_sigma * (dHmindE - dHdE)
        mis.guide["sigma_max_L1"] = L1
        mis.guide["sigma_max_Lmax"] = Lmax
        mis.guide["sigma_max_dHmindE"] = float(dHmindE)
        mis.guide["sigma_max_dHdE"] = dHdE
        mis.guide["sigma_max"] = sigma_max
        return sigma_max
        # return np.inf

    def update_path_angle_sgp(self, mis: Missile, tar: Missile = None, meta={}):
        """
        更新平稳滑翔所需的弹道倾角
        :param mis:
        :return:
        """
        rho = ATM.rho(mis.guide["h"])
        drho_dh = -0.00015 * rho
        v = mis.guide["v"]
        S = mis.guide["S"]
        m = mis.guide["m"]
        Rh = mis.guide["h"] + e.Re
        sigma_bsl = self.sigma_bsl(mis, tar, meta)
        alpha_bsl = self.attack_angle_plan(mis.guide['E'])
        # CL_bsl = cav_corridor.interp_CL_E(mis.guide["E"])
        # D_bsl = cav_corridor.interp_CD_E(mis.guide["E"]) * mis.guide["q_inf"]
        # 使用参考攻角与当前马赫数计算参考升阻系数
        CL_bsl, CD_bsl = mis.aero.CLCD(mis.ma, alpha_bsl)
        D_bsl = CD_bsl * mis.guide["q_inf"]
        dCL_dE = cav_corridor.interp_dCL_dE(mis.guide["E"])
        d1 = rho * v ** 2 * S * cos(sigma_bsl) / 2 / m * dCL_dE + 2 / Rh + CL_bsl * rho * S * cos(sigma_bsl) / m
        d2 = -CL_bsl * v ** 2 * S * cos(sigma_bsl) * drho_dh / 2 / e.g0 / m + 2 / Rh + CL_bsl * rho * S * cos(
            sigma_bsl) / m + v ** 2 / Rh ** 2 / e.g0
        mis.guide["gamma_sg"] = -D_bsl / m / e.g0 * d1 / d2

    def alpha_bsl(self, mis: Missile, tar: Missile = None, meta={}):
        return self.attack_angle_plan(mis.guide['E'])

    def sigma_bsl(self, mis: Missile, tar: Missile = None, meta={}):
        if self.guide_mode == 'descent_phase1' or self.guide_mode == 'descent_phase2':
            return 0
        if self.guide_mode == 'steady_glide_phase':
            return self.sigma_bsl_sgp(mis, tar, meta)
        if self.guide_mode == 'altitude_adjustment_phase':
            return self.sigma_bsl_aap(mis, tar, meta)

    def sigma_bsl_sgp(self, mis: Missile, tar: Missile = None, meta={}):
        L1_L = self.L1D(mis) / (mis.guide["CL"] / mis.guide["CD"])
        assert 1 >= L1_L >= -1, '超出射程范围'
        res = self.sgn * arccos(L1_L)
        if mis.guide["E"] > self.EBR1:
            return res
        else:
            return -res

    def sigma_bsl_aap(self, mis: Missile, tar: Missile = None, meta={}):
        gamma = mis.status.path_angle
        phi = mis.status.latitude
        psi = mis.status.heading_angle
        H = e.Re + mis.guide["h"]
        v = mis.guide["v"]
        delta_psi = mis.guide['delta_psi']
        s_go = mis.guide["s_go"]
        s_go_EBR2 = mis.guide["sgo_EBR2"]
        d_psi_LOS = mis.guide["v"] * cos(gamma) * sin(delta_psi) / s_go
        k_PN = 4 - 2 * s_go / s_go_EBR2

        delta_L1_m = e.omega_e ** 2 * H * cos(phi) ** 2 + 2 * v * e.omega_e * cos(phi) * sin(psi)
        delta_L2_m = -e.omega_e ** 2 * H * sin(phi) * cos(phi) * sin(psi) + 2 * v * e.omega_e * sin(phi)
        aL1 = e.g0 - v ** 2 / H - delta_L1_m
        aL2 = k_PN * d_psi_LOS * v * cos(gamma) - delta_L2_m
        mis.guide["sigma_aap_aL2"] = aL2
        mis.guide["sigma_aap_deltaL2m"] = delta_L2_m
        return -arctan(aL2 / aL1)

    def L1D(self, mis: Missile, E=None):
        if not E:
            E = mis.guide["E"]
        if E >= self.E_alpha:
            return mis.guide["L1D1"]
        else:
            return ((self.E_alpha - E) / (self.E_alpha - self.E_TAEM)) ** 2 * (
                    mis.guide["L1D2"] - mis.guide["L1D1"]) + mis.guide["L1D1"]

    def update_L1D(self, mis: Missile, tar: Missile = None, meta={}):
        # 1. 计算L1/D2
        L1D2 = self.L1D2(mis)
        mis.guide["L1D2"] = L1D2
        # 2. 计算L1/D1
        kxD1, kxD2 = self.kxD(mis, tar, meta)
        mis.guide["L1D1"] = (mis.guide["s_go"] - self.S_TAEM - kxD1 * L1D2) / (-kxD1 + kxD2)

    def update_EBR1(self, mis: Missile, tar: Missile = None, meta={}):
        print(f'更新EBR1,t={meta["t"]}')
        optimize.newton(lambda x: self.tempxCf(mis, x), self.EBR1, fprime=lambda x: self.tempdxCf(mis, x), tol=1e3)

    def tempxCf(self, mis, EBR1):
        self.EBR1 = EBR1
        return self.xCf(mis)

    def tempdxCf(self, mis, EBR1):
        return self.dxCf(mis)

    def xCf(self, mis: Missile):
        Rs = e.Re + cav_corridor.average_h_e
        E = mis.guide["E"]
        kh3 = mis.guide["kh3"]
        h3 = lambda x: (kh3[0] + kh3[1] * x + kh3[2] * x ** 2) / self.h_m(mis, x)
        delta_psi = mis.guide['delta_psi']
        integ = integrate.quad(
            lambda x: sin(self.xD(mis, self.E_TAEM, x) / e.Re) * self.L1D(mis, x) * h3(x) / (2 * x + 2 * e.mu / Rs)
            , E, self.E_TAEM)
        res = e.Re * delta_psi * sin(self.xD(mis, self.E_TAEM, E) / e.Re) + e.Re * integ[0] - self.sgn * e.Re * (
                self.F(mis, self.EBR1, E) - self.F(mis, self.EBR2, self.EBR1) + self.F(mis, self.E_TAEM, self.EBR2))
        return res

    def dxCf(self, mis: Missile):
        return -2 * self.sgn * e.Re * sin(self.xD(mis, self.E_TAEM, self.EBR1) / e.Re) * self.f2(mis, self.EBR1)

    def F(self, mis: Missile, x2, x1):
        res = integrate.quad(lambda x: sin(self.xD(mis, self.E_TAEM, x) / e.Re) * self.f2(mis, x), x1, x2)
        return res[0]

    def L2D(self, mis: Missile, E=None):
        if not E:
            E = mis.guide["E"]
        return abs(np.sqrt(cav_corridor.L2D_E(E) ** 2 - self.L1D(mis, E) ** 2))

    def f2(self, mis: Missile, E=None):
        if not E:
            E = mis.guide["E"]
        return self.L2D(mis, E) * (1 + (mis.guide["kh2"][0] + mis.guide["kh2"][1] * E) / self.h_m(mis, E)) / (
                2 * E + 2 * e.mu / (cav_corridor.average_h_e + e.Re))

    def xD(self, mis: Missile, E_end, E_start):
        """
        射程估计
        :param mis:
        :param E_end, E_start: E_end < E_start
        :return:
        """
        # assert 后面的元组不能加括号，否则恒为True
        assert E_start >= E_end, '起始能量应该大于终止能量'
        L1D1 = mis.guide["L1D1"]
        kh1 = mis.guide["kh1"]
        Rs = e.Re + cav_corridor.average_h_e
        temp1 = np.log((2 * E_end + e.mu / Rs) / (2 * E_start + e.mu / Rs))
        temp2 = 1 / (2 * E_end + e.mu / Rs) - 1 / (2 * E_start + e.mu / Rs)
        if self.E_alpha <= E_end:
            res = e.Re * L1D1 / 2 * (1 + kh1[1] / 2) * temp1 - e.Re * L1D1 / 2 * (
                    kh1[0] - e.mu * kh1[1] / 2 / Rs) * temp2
        elif self.E_alpha >= E_start:
            L1D2 = mis.guide["L1D2"]
            a0 = (L1D2 - L1D1) * self.E_alpha ** 2 / (self.E_alpha - self.E_TAEM) ** 2 + L1D1
            a1 = -2 * (L1D2 - L1D1) * self.E_alpha / (self.E_alpha - self.E_TAEM) ** 2
            a2 = (L1D2 - L1D1) / (self.E_alpha - self.E_TAEM) ** 2
            res = (1 + kh1[1] / 2) * e.Re * a2 / 4 * (E_end ** 2 - E_start ** 2) + e.Re / 2 * (
                    (1 + kh1[1] / 2) * a1 + (-1 - kh1[1] + Rs * kh1[0] / e.mu) * e.mu * a2 / 2 / Rs) * (E_end - E_start) \
                  + e.Re / 2 * ((1 + kh1[1] / 2) * a0 + (-1 - kh1[1] + Rs * kh1[0] / e.mu) * e.mu * a1 / 2 / Rs) * temp1 \
                  + e.Re / 2 * (e.mu ** 2 * (2 + 3 * kh1[1]) - 4 * e.mu * Rs * kh1[0]) / (8 * Rs ** 2) * a2 * temp1 \
                  - e.Re / 2 * (kh1[0] - e.mu * kh1[1] / 2 / Rs) * (
                          a0 - e.mu * a1 / 2 / Rs + e.mu ** 2 / 4 / Rs ** 2 * a2) * temp2
        else:
            res = self.xD(mis, self.E_alpha, E_start) + self.xD(mis, E_end, self.E_alpha)

        return res

    def kxD(self, mis: Missile, tar: Missile, meta={}):
        Re = e.Re
        kh1 = mis.guide["kh1"]
        Rs = cav_corridor.average_h_e + Re
        Ea, Et = self.E_alpha, self.E_TAEM
        kxd1 = -Re / 4 * (1 + kh1[1] / 2) * (Ea + Et) / (Ea - Et) \
               + Re / 2 / (Ea - Et) * (2 * Ea + e.mu / 2 / Rs + (Ea + e.mu / 2 / Rs) * kh1[1] - kh1[0] / 2) \
               + np.log((2 * Et + e.mu / Rs) / (2 * Ea + e.mu / Rs)) * Re * (2 * Ea * Rs + e.mu) / 2 / (Ea - Et) ** 2 * \
               ((2 * Ea * Rs + e.mu) / 4 / Rs ** 2 - kh1[0] / 2 / Rs + (2 * Ea * Rs + 3 * e.mu) / 8 / Rs ** 2 * kh1[1]) \
               - Re * (2 * Ea * Rs + e.mu) ** 2 / 8 / Rs ** 2 / (Ea - Et) ** 2 * \
               (kh1[0] - e.mu * kh1[1] / 2 / Rs) * (1 / (2 * Et + e.mu / Rs) - 1 / (2 * Ea + e.mu / Rs))
        log_ = (2 * Et + e.mu / Rs) / (2 * mis.guide["E"] + e.mu / Rs)
        kxd2 = Re / 2 * (1 + kh1[1] / 2) * np.log(log_) \
               - Re / 2 * (kh1[0] - e.mu * kh1[1] / 2 / Rs) * \
               (1 / (2 * Et + e.mu / Rs) - 1 / (2 * mis.guide["E"] + e.mu / Rs))
        return kxd1, kxd2

    def update_kh(self, mis: Missile, tar: Missile, meta={}):
        Rs = cav_corridor.average_h_e + e.Re
        m_phi = mis.status.latitude
        t_phi = tar.status.latitude
        E = mis.guide["E"]
        ET = self.E_TAEM
        ha = mis.guide["great_circle_heading"]
        # 以L/D * cos(σ)代表上一步的纵向升阻比
        hz1 = -2 * Rs * e.omega_e * mis.guide["v"] * cos(m_phi) * sin(ha[0]) - \
              Rs * e.omega_e ** 2 * (e.Re + mis.guide["h"]) * cos(m_phi) * \
              (cos(m_phi) - mis.guide["CL"] / mis.guide["CD"] * cos(mis.control["bank_angle"]) * sin(
                  m_phi) * cos(ha[0]))
        hz1_t = -2 * Rs * e.omega_e * self.V_TAEM * cos(t_phi) * sin(ha[1]) - \
                Rs * e.omega_e ** 2 * (e.Re + self.H_TAEM) * cos(t_phi) * \
                (cos(t_phi) - mis.guide["L2Dbsl_TAEM"] * sin(
                    t_phi) * cos(ha[1]))
        kh1 = (hz1_t * E - hz1 * ET) / (E - ET), \
              (hz1 - hz1_t) / (E - ET)

        hz2 = e.omega_e ** 2 * Rs * (e.Re + mis.guide["h"]) * mis.guide["CL"] / mis.guide["CD"] * cos(
            mis.control["bank_angle"]) * sin(m_phi) * cos(m_phi) * cos(ha[0])
        hz2_t = e.omega_e ** 2 * Rs * (e.Re + self.H_TAEM) * mis.guide["L2Dbsl_TAEM"] * sin(t_phi) * cos(
            t_phi) * cos(ha[1])

        kh2 = (hz2_t * E - hz2 * ET) / (E - ET), \
              (hz2 - hz2_t) / (E - ET)

        kh3 = -2 * e.omega_e * Rs * (self.V_TAEM * E - ET * mis.guide["v"]) * (sin(t_phi) * E - sin(m_phi) * ET) / (
                E - ET) ** 2, \
              -2 * e.omega_e * Rs * (self.V_TAEM * sin(m_phi) + mis.guide["v"] * sin(t_phi)) * (E + ET) / (E - ET) ** 2 \
              + 4 * e.omega_e * Rs * (self.V_TAEM * sin(t_phi) * E + mis.guide["v"] * sin(m_phi) * ET) / (E - ET) ** 2, \
              -2 * e.omega_e * Rs * (mis.guide["v"] - self.V_TAEM) * (sin(m_phi) - sin(t_phi)) / (E - ET) ** 2

        mis.guide["kh1"] = kh1
        mis.guide["kh2"] = kh2
        mis.guide["kh3"] = kh3

    def h_m(self, mis: Missile, E=None):
        if not E:
            E = mis.guide["E"]
        Rs = e.Re + cav_corridor.average_h_e
        return 2 * E + e.mu / Rs

    def L1D2(self, mis: Missile):
        # k_L/D * L/D_bsl_TAEM 取k_L/D=1
        return mis.guide["L2Dbsl_TAEM"]

    def s_go(self, mis: Missile, tar: Missile = None, meta={}):
        x_mis = ct.coordinate_transformation(0, lamb=mis.status.longitude, phi=mis.status.latitude)
        x_tar = ct.coordinate_transformation(0, lamb=tar.status.longitude, phi=tar.status.latitude)
        return e.Re * arccos(x_mis.dot(x_tar))

    def guide2control(self, mis: Missile, tar: Missile = None, meta={}):
        for item in self.control_param_list:
            mis.control[item] = mis.guide[item]
        mis.control['CL'], mis.control['CD'] = mis.aero.CLCD(mis.ma, mis.guide["attack_angle"])
        mis.control['L'] = mis.control['CL'] * mis.guide['q_inf']
        mis.control['D'] = mis.control['CD'] * mis.guide['q_inf']

    def end_guide(self, mis: Missile = Missile(), tar: Missile = Missile(), meta={}, flag=False):
        if self.end_flag:
            return True

        if mis.guide.get('s_go', np.inf) < self.S_TAEM:
            self.end_flag = True

        if mis.guide.get('E', np.inf) < (self.E_TAEM - 5E5) and not self.allow_update_param:
            self.end_flag = True

        if flag:
            self.end_flag = flag
        return self.end_flag
