import numpy as np

import settings as glbs
from entity.missile import Missile
from guidance.guide import Guidance
from simulation import TrajectorySimulation
from store.dataSave import DataSave
from utils.integral import Integral


class CustomSimulation(TrajectorySimulation):
    def __init__(self):
        super(CustomSimulation, self).__init__()

    def init(self, mis=Missile(), tar=Missile(), guide=Guidance(), integ=Integral(), db=DataSave()):
        self.mis = mis
        self.tar = tar
        self.guide = guide
        self.integral = integ
        self.db = db
        # 初始化导弹及目标状态
        self.mis.status.change_stat(glbs.MissileInitStatus)
        self.tar.status.change_stat(glbs.TargetInitStatus)
        # 初始化制导模型
        self.guide.init(self.mis, self.tar)

        self.is_online = False

    def step_len(self) -> float:
        if self.guide.accurate_mode:
            return pow(0.1, self.guide.accurate_mode)
        return 1

    def save_data(self) -> None:
        self.db_save_dict = {"global_t": self.t,
                             "E": self.mis.guide["E"]}
        self.db_save_dict.update(self.mis.status.status_dict())
        self.db_save_dict.update(self.mis.control)
        self.db_save_dict.update(self.from_mis_guide(getattr(glbs, 'GUIDE_SAVE_PARAM', [])))

        self.db.update(self.db_save_dict)

    def is_continue(self):
        return True

    def after_simulation(self):
        if self.is_online:
            return
        status = {
            '经度': np.rad2deg(self.mis.status.longitude),
            '纬度': np.rad2deg(self.mis.status.latitude),
            '时间': self.mis.status.t,
            '速度': self.mis.status.velocity,
            '高度': self.mis.status.height,
            'Δψ': self.mis.guide["delta_psi"]
        }
        print('导弹结束状态：')
        for k in status.keys():
            print(f'{k}: {status[k]}\n')
