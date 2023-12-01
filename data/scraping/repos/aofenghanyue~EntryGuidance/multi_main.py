# -*- coding: utf-8 -*-
# EditTime  : 2021-10-27 9:27
# Author    : Of yue
# File      : multi_main.py
# Intro     :

import pandas

import settings as glbs
from guidance.multiMissileGuideInstance import MultiMisGuideInstance
from core.multiMissileSimInstance import MultiMisSimInstance
from database.vehicleParams import CAVHParams
from dynamics.aerodynamic import AerodynamicCAVH
from dynamics.motionEquation import ME6D
from entity.missile import Missile
from store.dataSave import DataSave
from store.status import ME6DStatus, MissileStatus
from utils.integral import RungeKutta4

mis = [Missile(motion_equation=ME6D(),
              aerodynamic=AerodynamicCAVH(),
              params=CAVHParams(),
              status=ME6DStatus())]
# 2. 创建目标对象
tar = [Missile(status=MissileStatus())]
# 3. 创建制导模块
guide = MultiMisGuideInstance()
# 4. 创建积分模块
integral = RungeKutta4()
# 5. 创建数据存储模块
database = DataSave()

"""创建仿真模型"""
simulation = MultiMisSimInstance()
simulation.init(mis=mis, tar=tar, guide=guide, integ=integral, db=database)

"""开始仿真"""
simulation.simulation()

result = simulation.db[0].data
pass