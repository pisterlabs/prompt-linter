# -*- coding: utf-8 -*-
# EditTime  : 2021-09-27 19:13
# Author    : Of yue
# File      : MultiMissileSim.py
# Intro     :
from itertools import chain
from typing import List

from entity.missile import Missile
from guidance.MultiGuide import MultiMissileGuidance
from store.dataSave import DataSave
from utils.integral import Integral
import multiset as glbs


class MultiMissileSim:
    def __init__(self):
        # 最大迭代次数
        self.max_iter = getattr(glbs, "MAXITER", 0)
        # 当前迭代次数
        self.iter = 0

        # 最小积分步长
        self.min_h = getattr(glbs, "MIN_H", 1E-4)

        # 全局时间
        self.t = getattr(glbs, "t0", 0)

        # 参与仿真的导弹
        self.mis: List[Missile] = []
        self.tar: List[Missile] = []

        # 制导模块
        self.guide: MultiMissileGuidance = MultiMissileGuidance()
        self.guide_params_list = getattr(glbs, "ParamsToGuide", [])

        # 积分模块
        self.integral = Integral()

        # 数据存储模块
        self.db: List[DataSave] = []
        self.db_save_dict: dict = {}  # 每次存储值的临时字典
        self.init()

        # 其它
        # 是否为在线临时仿真
        self.is_main = True

    def init(self, mis=[], tar=[], guide=MultiMissileGuidance(), integ=Integral(), db=[]):
        """
        初始化导弹，目标，制导模式，积分，存储模块等信息，需重载
        :param mis: List[Missile]
        :param tar: List[Missile]
        :param guide: Guidance
        :param integ: Integral
        :param db: List[DataSave]
        :return:
        """
        pass

    def step_len(self):
        """
        计算每一步的积分步长，需重载
        :return:
        """
        return self.min_h

    def simulation(self):
        self._before_simulation()
        while self.next():
            self._after_one_step()
        self._after_simulation()

    def next(self):
        """
        单次仿真循环，当停止制导后自动结束
        :return:
        """
        if self._is_continue():
            if self.one_step_guide():
                # 此处应该先保存上一步的状态值，再改变状态
                self.after_guide()
                # 积分且更新状态
                self.one_step_integral()  # 包含了self.one_step_update_status()
                return True
        return False

    def one_step_guide(self):
        """
        1.产生控制指令
        :return: True代表单步制导执行成功
                False代表制导结束
        """
        # 产生传入制导模块的相关参数
        # 产生制导指令, 如果制导指令成功生成，则继续仿真
        return self.guide.one_step_guide(self.mis, self.tar, self.guide_params())

    def one_step_integral(self):
        """
        2.各个体单步积分
        :return:
        """
        for m in chain(self.mis, self.tar):
            self.one_step_integral_object(m)

    def one_step_integral_object(self, obj: Missile):
        # 如果时间累计大于步长，则制导一次
        if not obj.guide["guide_process"]:
            return False
        if obj.guide["end_guide"]:
            return False
        if obj.launched:
            # 如果导弹已发射则更新状态
            x, y0 = obj.status.x, obj.status.y

            temp_step_len = obj.guide.get("step_len", self.step_len())
            y_next, dy = self.integral.next_step(obj.equation.equation, x, y0, temp_step_len, obj.control, need_dy=True)
            x_next = x + temp_step_len
            if dy is not None:
                obj.guide["dy"] = dy

            # 3.各个体更新状态
            status_update_data = {k: v for k, v in zip(obj.status.integral_key, y_next)}
            status_update_data.update({obj.status.independent_key: x_next})
            obj.status.change_stat(status_update_data)

    def guide_params(self):
        """
        产生要传入制导模块的参数
        :return: None
        """
        temp_guide_params = {}
        for param in self.guide_params_list:
            if param == "self":
                temp_guide_params["simulation"] = self
            else:
                temp_guide_params[param] = getattr(self, param, None)
        return temp_guide_params

    def save_data(self) -> None:
        """
        存储数据
        :return: None
        """
        for index, (mis, db) in enumerate(zip(self.mis, self.db)):
            # 如果此仿真周期进行了制导，则保存数据
            if mis.guide["guide_process"] and not mis.guide["end_guide"]:
                db_save_dict = {"global_t": self.t}
                # 默认只存储时间、导弹状态和控制量
                db_save_dict.update(mis.status.status_dict())
                db_save_dict.update(mis.control)

                db.update(db_save_dict)

    def _is_continue(self) -> bool:
        """判断是否继续仿真"""
        if self.iter > self.max_iter:
            return False
        if not self.is_continue():
            return False
        return True

    def is_continue(self):
        return True

    def after_guide(self):
        """
        制导指令计算完成后进行
        可存储导弹及其控制指令
        :return:
        """
        self.save_data()

    def _after_one_step(self):
        """每个积分循环完成后需要进行的操作
        可以存储数据等，但一般存储数据在制导完成之后进行"""
        # 迭代次数+1
        self.iter += 1
        # 时间前进h
        self.t += self.min_h
        self.after_one_step()

    def after_one_step(self):
        pass

    def _before_simulation(self):
        print("仿真开始")
        self.before_simulation()

    def before_simulation(self):
        pass

    def _after_simulation(self):
        """仿真结束后需要做的事情
        例如将结果输出为图像"""
        print("仿真结束")
        self.after_simulation()
        pass

    def after_simulation(self):
        pass

    def from_mis_guide(self, mis: Missile, param_list: list):
        return {param: mis.guide.get(param, None)
                for param in param_list}
