import settings as glbs
from entity.missile import Missile
from guidance.guide import Guidance
from store.dataSave import DataSave
from utils.integral import Integral


class TrajectorySimulation:
    def __init__(self):
        # 最大迭代次数
        self.max_iter = getattr(glbs, "MAXITER", 0)
        # 当前迭代次数
        self.iter = 0

        # 当前积分步长
        self.h = 0

        # 全局时间
        self.t = getattr(glbs, "t0", 0)
        # 导弹与其对应目标
        self.mis: Missile = Missile()
        self.tar: Missile = Missile()

        # 制导模块
        self.guide: Guidance = Guidance()
        self.guide_params_list = getattr(glbs, "ParamsToGuide", [])
        self.guide_params: dict = {}

        # 积分模块
        self.integral = Integral()

        # 数据存储模块 可以是列表，对应不同的导弹与目标
        self.db = DataSave()
        self.db_save_dict: dict = {}  # 每次存储值的临时字典
        self.init()

    def init(self, mis=Missile(), tar=Missile(), guide=Guidance(), integ=Integral(), db=DataSave()):
        """
        初始化导弹，目标，制导模式，积分，存储模块等信息
        :return:
        """
        pass

    def step_len(self) -> float:
        """
        计算每一步的积分步长
        :return: 当前步积分步长
        """
        pass

    def simulation(self):
        self._before_simulation()
        while self.next():
            self._after_one_step()
        self._after_simulation()

    def next(self):
        """
        一个循环
        :return:
        """
        if self._is_continue():
            self.h = self.step_len()
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
        # 产生制导相关参数
        self.gen_guide_params()
        # 产生制导指令
        if self.guide.one_step_guide(self.mis, self.tar, self.guide_params):
            return True
        else:
            return False

    def one_step_integral(self):
        """
        2.各个体单步积分
        :return: 
        """
        self.one_step_integral_object(self.mis)
        self.one_step_integral_object(self.tar)

    def one_step_integral_object(self, obj: Missile):
        if obj.launched:
            # 如果导弹已发射则更新状态
            x, y0 = obj.status.x, obj.status.y
            y_next, dy = self.integral.next_step(obj.equation.equation, x, y0, self.h, obj.control, need_dy=True)
            x_next = x + self.h
            if dy is not None:
                obj.guide["dy"] = dy

            # 3.各个体更新状态
            status_update_data = {k: v for k, v in zip(obj.status.integral_key, y_next)}
            status_update_data.update({obj.status.independent_key: x_next})
            obj.status.change_stat(status_update_data)

    def gen_guide_params(self):
        """
        产生要传入制导模块的参数
        :return: None
        """
        for param in self.guide_params_list:
            if param == "self":
                self.guide_params["simulation"] = self
            else:
                self.guide_params[param] = getattr(self, param, None)

    def save_data(self) -> None:
        """
        存储数据
        :return: None
        """
        self.db_save_dict = {"global_t": self.t}
        # 默认只存储时间、导弹状态和控制量
        self.db_save_dict.update(self.mis.status.status_dict())
        self.db_save_dict.update(self.mis.control)

        self.db.update(self.db_save_dict)

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
        self.t += self.h
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

    def from_mis_guide(self, param_list: list):
        return {param: self.mis.guide.get(param, None) for param in param_list}
