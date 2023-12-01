# -*- coding: utf-8 -*-
# EditTime  : 2021-09-29 20:09
# Author    : Of yue
# File      : multiMissileSimInstance.py
# Intro     :

import numpy as np

import multiset as glbs
from entity.missile import Missile
from guidance.MultiGuide import MultiMissileGuidance
from core.MultiMissileSim import MultiMissileSim
from store.dataSave import DataSave
from utils.integral import Integral


class MultiMisSimInstance(MultiMissileSim):
    def __init__(self):
        super(MultiMisSimInstance, self).__init__()

    def init(self, mis=[], tar=[], guide=MultiMissileGuidance(), integ=Integral(), db=[]):
        self.mis = mis
        self.tar = tar
        self.guide = guide
        self.integral = integ
        self.db = [db for _ in mis]
        # 初始化导弹及目标状态
        for m, t, status in zip(self.mis, self.tar, glbs.StatusParams):
            m.status.change_stat(status["MissileInitStatus"])
            t.status.change_stat(status["TargetInitStatus"])
        # 初始化制导模型
        self.guide.init(self.mis, self.tar, self.guide_params())

    def save_data(self) -> None:
        for index, (mis, db) in enumerate(zip(self.mis, self.db)):
            # 如果此仿真周期进行了制导，则保存数据
            if mis.guide["guide_process"] and not mis.guide["end_guide"]:
                db_save_dict = {"global_t": self.t, "E": mis.guide["E"]}
                db_save_dict.update(mis.status.status_dict())
                db_save_dict.update(mis.control)
                db_save_dict.update(self.from_mis_guide(mis, getattr(glbs, 'GUIDE_SAVE_PARAM', [])))

                db.update(db_save_dict)
