# -*- coding: utf-8 -*-

import sys
import csv
import os
import io
from functools import partial
import time

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QMessageBox, QFileDialog, QComboBox, QSlider, \
    QCheckBox, QGroupBox, QLineEdit, QTextEdit, QTableView, QHeaderView, QAbstractItemView, QTableWidgetItem, QMenu, \
    QAction, QFormLayout, QScrollArea, QSizePolicy, QGridLayout, QProgressBar, QVBoxLayout
from PyQt5.QtGui import QPixmap, QIcon, QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt, QEvent, QSize, QPoint

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# 获取exe根目录
base_path: str
base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

QApplication.addLibraryPath(base_path + "/plugins")

# 读取出力数据
province_list = ["湖南", "湖北", "河南", "河北", "山西", "天津", "北京", "山东", "江苏", "浙江", "上海", "安徽",
                 "江西", "福建"]


# 绘制进入页面
class EntryWindow(QWidget):

    # 界面初始化
    def __init__(self):
        super().__init__()

        # 设置界面标题
        self.setWindowTitle("储能优化程序ver3.0")
        # 设置页面图标
        self.setWindowIcon(QIcon(base_path + "/images/软件图标.png"))
        # 设置界面大小
        self.resize(1280, 720)
        # 设置背景图片
        self.background_image = QLabel(self)
        self.background_image.setPixmap(QPixmap(base_path + "/images/进入界面（2023-6-23）.png"))

        # 设置导入按钮
        self.import_button = QPushButton(self)
        self.import_button.setText("导入")
        self.import_button.move(20, 30)
        self.import_button.setFixedHeight(30)
        self.import_button.setFixedWidth(100)
        self.import_button.setStyleSheet("font:16px Microsoft YaHei; color:white; background-color:rgb(47, 52, 73); "
                                         "border-radius: 3px")
        # 在按钮正下方弹出菜单
        self.import_button.clicked.connect(lambda: self.import_menu.popup(
            self.mapToGlobal(QPoint(self.import_button.pos().x(), self.import_button.pos().y() + 30))))

        # 设置导入菜单
        self.import_menu = QMenu(self)
        self.import_menu.setStyleSheet("font:16px Microsoft YaHei; color:white; background-color:rgb(47, 52, 73); "
                                       "border-radius: 3px")

        # 设置导入潮流缺额数据菜单
        self.import_loss = QMenu(self)
        self.import_loss.setTitle("导入潮流缺额数据")
        self.import_loss.setStyleSheet("font:16px Microsoft YaHei; color:white; background-color:rgb(47, 52, 73); "
                                       "border-radius: 3px")
        for i in province_list:
            self.import_loss.addAction(i, partial(self.import_loss_data, i))

        self.import_menu.addMenu(self.import_loss)

        # 设置导入储能电站数据菜单
        self.import_station = QMenu(self)
        self.import_station.setTitle("导入储能电站数据")
        self.import_station.setStyleSheet("font:16px Microsoft YaHei; color:white; background-color:rgb(47, 52, 73); "
                                          "border-radius: 3px")
        for i in province_list:
            self.import_station.addAction(i, partial(self.import_station_data, i))
        self.import_menu.addMenu(self.import_station)

        # 默认文件路径
        for i in province_list:
            exec(f"self.{i}_loss_path = 'data/{i}_loss.csv'")
            exec(f"self.{i}_station_path = 'data/{i}_station.csv'")

        # 设置算法选择框
        self.algorithm_combobox = QComboBox(self)
        self.algorithm_combobox.move(150, 30)
        self.algorithm_combobox.setFixedHeight(30)
        self.algorithm_combobox.setFixedWidth(160)
        self.algorithm_combobox.setStyleSheet("font:16px Microsoft YaHei; color:white; background-color:rgb(47, 52, "
                                              "73); border-radius: 3px")
        self.algorithm_combobox.addItem("线性规划")
        self.algorithm_combobox.addItem("遗传算法")
        self.algorithm_combobox.addItem("粒子群算法")
        self.algorithm_combobox.addItem("差分进化算法")
        self.algorithm_combobox.addItem("模拟退火算法")
        self.algorithm_combobox.addItem("人工鱼群算法")
        self.algorithm_combobox.addItem("比较取优")

        # 设置比较窗口，包含以上几个算法的CheckBox
        class CompareWindow(QWidget):

            def __init__(self):
                super().__init__()

                self.setWindowTitle("比较取优")
                self.setFixedSize(300, 300)
                self.setWindowIcon(QIcon(base_path + "/images/软件图标.png"))

                layout = QVBoxLayout(self)
                layout.setSpacing(10)

                self.checkbox_linear = QCheckBox("线性规划", self)
                self.checkbox_linear.setFixedHeight(30)
                self.checkbox_linear.setStyleSheet("font:18px Microsoft YaHei; border: 1px solid rgb(173, 173, 173);"
                                                   " border-radius: 3px")
                self.checkbox_linear.setChecked(True)
                layout.addWidget(self.checkbox_linear)

                self.checkbox_ga = QCheckBox("遗传算法", self)
                self.checkbox_ga.setStyleSheet("font:18px Microsoft YaHei; border: 1px solid rgb(173, 173, 173);"
                                               " border-radius: 3px")
                self.checkbox_ga.setChecked(True)
                layout.addWidget(self.checkbox_ga)

                self.checkbox_pso = QCheckBox("粒子群算法", self)
                self.checkbox_pso.setStyleSheet("font:18px Microsoft YaHei; border: 1px solid rgb(173, 173, 173);"
                                                " border-radius: 3px")
                self.checkbox_pso.setChecked(True)
                layout.addWidget(self.checkbox_pso)

                self.checkbox_de = QCheckBox("差分进化算法", self)
                self.checkbox_de.setStyleSheet("font:18px Microsoft YaHei; border: 1px solid rgb(173, 173, 173);"
                                               " border-radius: 3px")
                self.checkbox_de.setChecked(True)
                layout.addWidget(self.checkbox_de)

                self.checkbox_sa = QCheckBox("模拟退火算法", self)
                self.checkbox_sa.setStyleSheet("font:18px Microsoft YaHei; border: 1px solid rgb(173, 173, 173);"
                                               " border-radius: 3px")
                self.checkbox_sa.setChecked(True)
                layout.addWidget(self.checkbox_sa)

                self.checkbox_afsa = QCheckBox("人工鱼群算法", self)
                self.checkbox_afsa.setStyleSheet("font:18px Microsoft YaHei; border: 1px solid rgb(173, 173, 173);"
                                                 " border-radius: 3px")
                self.checkbox_afsa.setChecked(True)
                layout.addWidget(self.checkbox_afsa)

                self.button_compare = QPushButton(self)
                self.button_compare.setText("确定")
                self.button_compare.setStyleSheet("font:18px Microsoft YaHei; border: 1px solid rgb(173, 173, 173);"
                                                  " border-radius: 3px")
                self.button_compare.clicked.connect(lambda: self.hide())
                layout.addWidget(self.button_compare)

        self.compare_window = CompareWindow()
        self.algorithm_combobox.currentIndexChanged.connect(lambda: self.algorithm_select())

        # 设置AI交互窗口按钮
        self.button_ai = QPushButton(self)
        self.button_ai.setText("AI交互")
        self.button_ai.move(340, 30)
        self.button_ai.setFixedHeight(30)
        self.button_ai.setFixedWidth(80)
        self.button_ai.setStyleSheet("color:white; background-color:rgb(47, 52, 73); border-radius: 3px")
        self.button_ai.clicked.connect(lambda: self.ai_window.show())

        class AIWindow(QWidget):

            def __init__(self):
                super().__init__()

                self.setWindowTitle("AI交互窗口")
                self.setFixedSize(800, 600)

                self.setWindowIcon(QIcon(base_path + "/images/软件图标.png"))

                self.input_text = QLineEdit(self)
                self.input_text.move(20, 20)
                self.input_text.setFixedWidth(760)
                self.input_text.setStyleSheet("font:16px Microsoft YaHei; color:black;")

                self.combobox_prompt = QComboBox(self)
                self.combobox_prompt.move(40, 63)
                self.combobox_prompt.setFixedWidth(150)
                self.combobox_prompt.setStyleSheet("font:16px Microsoft YaHei; color:black;")
                self.combobox_prompt.addItem("AI参数配置")
                self.combobox_prompt.addItem("ChatGPT聊天")

                self.combobox_model = QComboBox(self)
                self.combobox_model.move(440, 63)
                self.combobox_model.setFixedWidth(200)
                self.combobox_model.setStyleSheet("font:16px Microsoft YaHei; color:black;")
                self.combobox_model.addItem("gpt-3.5-turbo")
                self.combobox_model.addItem("gpt-3.5-turbo-16k")
                self.combobox_model.addItem("gpt-4")
                self.combobox_model.addItem("gpt-4-32k")
                self.combobox_model.addItem("text-davinci-003")
                self.combobox_model.addItem("text-embedding-ada-002")

                self.button_send = QPushButton("发送", self)
                self.button_send.move(660, 61)
                self.button_send.setFixedWidth(120)
                self.button_send.setStyleSheet("font:16px Microsoft YaHei; color:black;")
                self.button_send.setShortcut("ctrl+return")

                self.output_text = QTextEdit(self)
                self.output_text.move(20, 100)
                self.output_text.setFixedWidth(760)
                self.output_text.setFixedHeight(480)
                self.output_text.setStyleSheet("font:18px Microsoft YaHei;"
                                               "color:black;")
                self.output_text.setReadOnly(True)
                self.output_text.setStyleSheet("background-image:url(images/AI背景);")

                self.button_send.clicked.connect(lambda: self.AI_connect(self.combobox_model.currentText()))

            def AI_connect(self, model):
                import OpenAIConnection
                if self.combobox_prompt.currentText() == "AI参数配置":
                    self.output_text.append("用户：\n" + self.input_text.text() + "\n--------------------------")
                    self.output_text.append(
                        "AI：\n" + OpenAIConnection.chat(self.input_text.text(), model) + "\n--------------------------")
                    app.processEvents()
                    time.sleep(0.1)
                else:
                    self.output_text.append("用户：\n" + self.input_text.text() + "\n--------------------------")
                    self.output_text.append(
                        "AI：\n" + OpenAIConnection.chat(self.input_text.text(), model) + "\n--------------------------")
                    app.processEvents()
                    time.sleep(0.1)

        self.ai_window = AIWindow()

        # 设置AI算法参数配置窗口按钮
        self.button_argument = QPushButton(self)
        self.button_argument.setText("参数配置")
        self.button_argument.move(980, 30)
        self.button_argument.setFixedHeight(30)
        self.button_argument.setFixedWidth(80)
        self.button_argument.setStyleSheet("color:white; background-color:rgb(47, 52, 73); border-radius: 3px")
        self.button_argument.clicked.connect(lambda: self.argument_window.show())

        class ArgumentsWindow(QWidget):

            def __init__(self):
                super().__init__()

                self.setWindowTitle("AI算法参数配置")
                self.setFixedSize(800, 400)

                self.setWindowIcon(QIcon(base_path + "/images/软件图标.png"))

                # 使用gridlayout布局，每两行为一个算法，第一行为算法和参数名称，第二行为参数值
                self.gridlayout = QGridLayout(self)
                self.gridlayout.setSpacing(10)

                self.gridlayout.addWidget(QLabel("遗传算法"), 0, 0)
                self.gridlayout.addWidget(QLabel("种群规模"), 0, 1)
                self.gridlayout.addWidget(QLabel("迭代次数"), 0, 2)
                self.gridlayout.addWidget(QLabel("变异概率"), 0, 3)
                self.gridlayout.addWidget(QLabel("最高精度"), 0, 4)

                self.gridlayout.addWidget(QLabel(" "), 1, 0)
                self.gridlayout.addWidget(QTextEdit("50"), 1, 1)
                self.gridlayout.addWidget(QTextEdit("100"), 1, 2)
                self.gridlayout.addWidget(QTextEdit("0.001"), 1, 3)
                self.gridlayout.addWidget(QTextEdit("1000"), 1, 4)

                self.gridlayout.addWidget(QLabel("粒子群算法"), 2, 0)
                self.gridlayout.addWidget(QLabel("种群规模"), 2, 1)
                self.gridlayout.addWidget(QLabel("迭代次数"), 2, 2)
                self.gridlayout.addWidget(QLabel("惯性权重"), 2, 3)
                self.gridlayout.addWidget(QLabel("个体记忆"), 2, 4)
                self.gridlayout.addWidget(QLabel("集体记忆"), 2, 5)

                self.gridlayout.addWidget(QLabel(" "), 3, 0)
                self.gridlayout.addWidget(QTextEdit("50"), 3, 1)
                self.gridlayout.addWidget(QTextEdit("100"), 3, 2)
                self.gridlayout.addWidget(QTextEdit("0.8"), 3, 3)
                self.gridlayout.addWidget(QTextEdit("0.5"), 3, 4)
                self.gridlayout.addWidget(QTextEdit("0.5"), 3, 5)

                self.gridlayout.addWidget(QLabel("差分进化算法"), 4, 0)
                self.gridlayout.addWidget(QLabel("种群规模"), 4, 1)
                self.gridlayout.addWidget(QLabel("迭代次数"), 4, 2)
                self.gridlayout.addWidget(QLabel("变异概率"), 4, 3)
                self.gridlayout.addWidget(QLabel("变异系数"), 4, 4)

                self.gridlayout.addWidget(QLabel(""), 5, 0)
                self.gridlayout.addWidget(QTextEdit("50"), 5, 1)
                self.gridlayout.addWidget(QTextEdit("100"), 5, 2)
                self.gridlayout.addWidget(QTextEdit("0.001"), 5, 3)
                self.gridlayout.addWidget(QTextEdit("0.5"), 5, 4)

                self.gridlayout.addWidget(QLabel("模拟退火算法"), 6, 0)
                self.gridlayout.addWidget(QLabel("初始温度"), 6, 1)
                self.gridlayout.addWidget(QLabel("最小温度"), 6, 2)
                self.gridlayout.addWidget(QLabel("链长"), 6, 3)
                self.gridlayout.addWidget(QLabel("冷却耗时"), 6, 4)

                self.gridlayout.addWidget(QLabel(" "), 7, 0)
                self.gridlayout.addWidget(QTextEdit("100"), 7, 1)
                self.gridlayout.addWidget(QTextEdit("1e-7"), 7, 2)
                self.gridlayout.addWidget(QTextEdit("300"), 7, 3)
                self.gridlayout.addWidget(QTextEdit("150"), 7, 4)

                self.gridlayout.addWidget(QLabel("人工鱼群算法"), 8, 0)
                self.gridlayout.addWidget(QLabel("种群规模"), 8, 1)
                self.gridlayout.addWidget(QLabel("迭代次数"), 8, 2)
                self.gridlayout.addWidget(QLabel("最大尝试捕食次数"), 8, 3)
                self.gridlayout.addWidget(QLabel("单步位移比例"), 8, 4)
                self.gridlayout.addWidget(QLabel("鱼群最大感知范围"), 8, 5)
                self.gridlayout.addWidget(QLabel("鱼群感知范围衰减系数"), 10, 1)
                self.gridlayout.addWidget(QLabel("拥挤度阈值"), 10, 2)

                self.gridlayout.addWidget(QLabel(" "), 9, 0)
                self.gridlayout.addWidget(QTextEdit("50"), 9, 1)
                self.gridlayout.addWidget(QTextEdit("100"), 9, 2)
                self.gridlayout.addWidget(QTextEdit("100"), 9, 3)
                self.gridlayout.addWidget(QTextEdit("0.5"), 9, 4)
                self.gridlayout.addWidget(QTextEdit("0.3"), 9, 5)
                self.gridlayout.addWidget(QTextEdit("0.98"), 11, 1)
                self.gridlayout.addWidget(QTextEdit("0.5"), 11, 2)

        # 设置AI算法参数配置界面
        self.argument_window = ArgumentsWindow()

        # 设置电站信息按钮
        self.button_station = QPushButton(self)
        self.button_station.setText("电站信息")
        self.button_station.move(1080, 30)
        self.button_station.setFixedHeight(30)
        self.button_station.setFixedWidth(80)
        self.button_station.setStyleSheet("color:white; background-color:rgb(47, 52, 73); border-radius: 3px")
        self.station_window0 = StationWindow("湖南")
        self.button_station.clicked.connect(lambda: self.station_window0.show())

        # 设置省份按钮
        for i in province_list:
            exec(f"self.button{province_list.index(i)}_province = QPushButton(self)")
            exec(f"self.button{province_list.index(i)}_province.setText(i)")
            # 确定按钮位置，在(20,  130 + 30 * index)处
            exec(f"self.button{province_list.index(i)}_province.move(20, {130 + 30 * province_list.index(i)})")
            exec(f"self.button{province_list.index(i)}_province.setFixedHeight(28)")
            exec(f"self.button{province_list.index(i)}_province.setFixedWidth(90)")
            exec(f"self.button{province_list.index(i)}_province.setStyleSheet(\"color:white; background-color:rgb(47, "
                 f"52, 73); border-radius: 3px\")")
            exec(f"self.button{province_list.index(i)}_province.clicked.connect(partial(self.calculate, "
                 f"{province_list.index(i)}))")
        self.method = None
        self.calculate_arguments = None

        self.progress_window = None

        self.power_list = None
        self.H2_power_list = None
        self.ele_power_list = None
        self.water_power_list = None
        self.cost = None

        # 设置时间轴
        self.time_slider = QSlider(Qt.Horizontal, self)
        self.time_slider.setFixedWidth(180)
        self.time_slider.setRange(0, 95)
        self.time_slider.setSingleStep(1)
        self.time_slider.setStyleSheet("QSlider::handle:horizontal{background-color:#3f4459; \
                                      width: 15px; margin: -7px -7px -7px -7px; border-radius: 3px;}")
        self.time_slider.move(140, 607)

    def import_loss_data(self, province):
        exec(
            f"{province}_file, _ = QFileDialog.getOpenFileName(self, '打开潮流缺额数据（{province}）', base_path,"
            f" 'CSV Files (*.csv)')")
        exec(f"self.{province}_loss_path = {province}_file")
        try:
            exec(f"self.ResultWindow{province_list.index(province)}.close()")
        except AttributeError:
            pass

    def import_station_data(self, province):
        exec(
            f"{province}_file, _ = QFileDialog.getOpenFileName(self, '打开储能电站数据（{province}）', base_path,"
            f" 'CSV Files (*.csv)')")
        exec(f"self.{province}_station_path = {province}_file")
        try:
            exec(f"self.ResultWindow{province_list.index(province)}.close()")
        except AttributeError:
            pass

    def algorithm_select(self):
        if self.algorithm_combobox.currentText() == "线性规划":
            pass
        elif self.algorithm_combobox.currentText() == "遗传算法":
            self.method = "GA_planning"
            self.calculate_arguments = {
                "size_pop": int(self.argument_window.gridlayout.itemAtPosition(1, 1).widget().toPlainText()),
                "max_iter": int(self.argument_window.gridlayout.itemAtPosition(1, 2).widget().toPlainText()),
                "prob_mut": float(self.argument_window.gridlayout.itemAtPosition(1, 3).widget().toPlainText()),
                "precision": float(self.argument_window.gridlayout.itemAtPosition(1, 4).widget().toPlainText())
            }

        elif self.algorithm_combobox.currentText() == "粒子群算法":
            self.method = "PSO_planning"
            self.calculate_arguments = {
                "size_pop": int(self.argument_window.gridlayout.itemAtPosition(3, 1).widget().toPlainText()),
                "max_iter": int(self.argument_window.gridlayout.itemAtPosition(3, 2).widget().toPlainText()),
                "w": float(self.argument_window.gridlayout.itemAtPosition(3, 3).widget().toPlainText()),
                "c1": float(self.argument_window.gridlayout.itemAtPosition(3, 4).widget().toPlainText()),
                "c2": float(self.argument_window.gridlayout.itemAtPosition(3, 5).widget().toPlainText())
            }

        elif self.algorithm_combobox.currentText() == "差分进化算法":
            self.method = "DE_planning"
            self.calculate_arguments = {
                "size_pop": int(self.argument_window.gridlayout.itemAtPosition(5, 1).widget().toPlainText()),
                "max_iter": int(self.argument_window.gridlayout.itemAtPosition(5, 2).widget().toPlainText()),
                "prob_mut": float(self.argument_window.gridlayout.itemAtPosition(5, 3).widget().toPlainText()),
                "F": float(self.argument_window.gridlayout.itemAtPosition(5, 4).widget().toPlainText())
            }
        elif self.algorithm_combobox.currentText() == "模拟退火算法":
            self.method = "SA_planning"
            self.calculate_arguments = {
                "T_max": int(self.argument_window.gridlayout.itemAtPosition(7, 1).widget().toPlainText()),
                "T_min": float(self.argument_window.gridlayout.itemAtPosition(7, 2).widget().toPlainText()),
                "L": int(self.argument_window.gridlayout.itemAtPosition(7, 3).widget().toPlainText()),
                "max_stay_counter": float(self.argument_window.gridlayout.itemAtPosition(7, 4).widget().toPlainText()),
            }
        elif self.algorithm_combobox.currentText() == "人工鱼群算法":
            self.method = "AFSA_planning"
            self.calculate_arguments = {
                "size_pop": int(self.argument_window.gridlayout.itemAtPosition(9, 1).widget().toPlainText()),
                "max_iter": int(self.argument_window.gridlayout.itemAtPosition(9, 2).widget().toPlainText()),
                "max_try_num": int(self.argument_window.gridlayout.itemAtPosition(9, 3).widget().toPlainText()),
                "step": float(self.argument_window.gridlayout.itemAtPosition(9, 4).widget().toPlainText()),
                "visual": float(self.argument_window.gridlayout.itemAtPosition(9, 5).widget().toPlainText()),
                "q": float(self.argument_window.gridlayout.itemAtPosition(11, 1).widget().toPlainText()),
                "delta": float(self.argument_window.gridlayout.itemAtPosition(11, 2).widget().toPlainText())
            }
        else:
            self.compare_window.show()
            self.method = "GA_planning"
            self.calculate_arguments = {
                "size_pop": int(self.argument_window.gridlayout.itemAtPosition(1, 1).widget().toPlainText()),
                "max_iter": int(self.argument_window.gridlayout.itemAtPosition(1, 2).widget().toPlainText()),
                "prob_mut": float(self.argument_window.gridlayout.itemAtPosition(1, 3).widget().toPlainText()),
                "precision": float(self.argument_window.gridlayout.itemAtPosition(1, 4).widget().toPlainText())
            }

    def calculate(self, num):
        try:
            if eval(f"self.result_window{int(num)}.method == self.method"):
                # 弹出运算结果界面
                exec(f"self.result_window{int(num)}.show()")
                self.hide()
            else:
                raise AttributeError
        # 如果运算结果界面不存在，则创建
        except AttributeError:
            # 弹出进度条界面
            class ProgressBarWindow(QWidget):

                def __init__(self, entry_window):
                    super().__init__()

                    layout = QVBoxLayout(self)

                    self.setWindowTitle("计算中...")
                    self.setFixedSize(300, 100)

                    self.setWindowIcon(QIcon(base_path + "/images/软件图标.png"))

                    self.progress_bar = QProgressBar(self)
                    self.progress_bar.setStyleSheet("QProgressBar{border: 1px solid rgb(173, 173, 173);"
                                                    " border-radius: 3px; text-align:center}"
                                                    "QProgressBar::chunk{background-color:rgb(47, 52, 73);"
                                                    " border-radius: 3px;}")

                    try:
                        self.progress_bar.setRange(0, entry_window.calculate_arguments["max_iter"])
                    except TypeError and KeyError:
                        self.progress_bar.setRange(0, 100)
                    self.progress_bar.setValue(0)
                    layout.addWidget(self.progress_bar)

                    self.label = QLabel(self)
                    self.label.setStyleSheet("font:16px Microsoft YaHei; color:black;")
                    layout.addWidget(self.label)

                    self.setLayout(layout)

            self.progress_window = ProgressBarWindow(self)

            if self.calculate_arguments:
                try:
                    # 调用后端函数
                    import algorithm
                    loss_list = algorithm.loss_read(eval(f"self.{province_list[int(num)]}_loss_path"))
                    station_list = algorithm.station_read(eval(f"self.{province_list[int(num)]}_station_path"))
                    ai_planing = algorithm.AI_planing(loss_list, station_list)
                    self.progress_window.show()
                    try:
                        for i in range(self.calculate_arguments["max_iter"]):
                            ai_planing.calculate(eval(f"ai_planing.{self.method}"), 1, **self.calculate_arguments)
                            self.progress_window.progress_bar.setValue(i + 1)
                            self.progress_window.label.setText(f"当前解：{ai_planing.best_y[0][0]}")
                            app.processEvents()
                            time.sleep(0.1)
                    except KeyError:
                        ai_planing.calculate(eval(f"ai_planing.{self.method}"), 1, **self.calculate_arguments)
                    self.power_list = ai_planing.best_x
                    self.H2_power_list = ai_planing.best_H2_power
                    self.ele_power_list = ai_planing.best_ele_power
                    self.water_power_list = ai_planing.best_water_power
                    self.cost = ai_planing.best_y
                    self.progress_window.close()
                    exec(f"self.result_window{int(num)} = ResultWindow(self, self.method, {int(num)})")
                    self.hide()
                    exec(f"self.result_window{int(num)}.show()")
                except FileNotFoundError:
                    QMessageBox.critical(self, "错误", "请先导入电站信息！", QMessageBox.Ok, QMessageBox.Ok)

            else:
                try:
                    with open(f"data/{province_list[num]}.csv", "r", encoding="utf-8") as f:
                        data = csv.reader(f)
                        data = list(data)
                    self.progress_window.show()
                    # csv结构为：每行代表一个电站，每列代表一个时间段，每个元素代表该时间段的电出力
                    # 在第一列的最后一行后面还有一个元素，代表总成本
                    for i in range(len(data)):
                        for j in range(len(data[i])):
                            try:
                                data[i][j] = float(data[i][j])
                            except ValueError:
                                pass
                    if num == 0:
                        self.power_list = data[0:15]
                        self.H2_power_list = self.power_list[0:2]
                        self.ele_power_list = self.power_list[2:14]
                        self.water_power_list = [self.power_list[14]]
                        self.cost = [[data[15][0]]]
                    if num == 1:
                        self.power_list = data[0:13]
                        self.H2_power_list = self.power_list[0:2]
                        self.ele_power_list = self.power_list[2:12]
                        self.water_power_list = [self.power_list[12]]
                        self.cost = [[data[13][0]]]
                    for i in range(50):
                        self.progress_window.progress_bar.setValue(2 * i + 2)
                        app.processEvents()
                        time.sleep(0.5)
                    self.progress_window.close()
                    exec(f"self.result_window{int(num)} = ResultWindow(self, self.method, {int(num)})")
                    self.hide()
                    exec(f"self.result_window{int(num)}.show()")
                except FileNotFoundError:
                    QMessageBox.critical(self, "错误", "请先导入电站信息！", QMessageBox.Ok, QMessageBox.Ok)

    # 关闭窗口时弹出提示框
    def closeEvent(self, event):
        reply = QMessageBox.question(self, '提示', '是否要退出程序？', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


class StationWindow(QWidget):

    def __init__(self, province):
        super().__init__()
        self.province = province

        self.setWindowTitle("电站管理窗口")
        self.setFixedSize(800, 600)

        self.setWindowIcon(QIcon(base_path + "/images/软件图标.png"))

        self.label_province = QLabel(self)
        self.label_province.setText("当前省份：")
        self.label_province.move(20, 20)
        self.label_province.setFixedHeight(30)
        self.label_province.setFixedWidth(120)
        self.label_province.setStyleSheet("font:14px Microsoft YaHei; color:black;")

        self.combobox_province = QComboBox(self)
        self.combobox_province.move(100, 22)
        self.combobox_province.setFixedWidth(120)
        self.combobox_province.setStyleSheet("font:14px Microsoft YaHei; color:black;")
        self.combobox_province.addItems(province_list)
        self.combobox_province.setCurrentIndex(province_list.index(province))
        self.combobox_province.currentIndexChanged.connect(self.change_province)

        self.button_station_new = QPushButton(self)
        self.button_station_new.setText("新建电站(N)")
        self.button_station_new.move(300, 20)
        self.button_station_new.setFixedHeight(30)
        self.button_station_new.setFixedWidth(120)
        self.button_station_new.setStyleSheet("font:14px Microsoft YaHei; background-color: rgb(225, 225, 225); "
                                              "border: 1px solid rgb(173, 173, 173); border-radius: 3px")
        self.new_window0 = self.new_window(self)
        self.button_station_new.clicked.connect(lambda: self.new_window0.show())
        self.button_station_new.setShortcut("ctrl+N")

        self.button_station_save = QPushButton(self)
        self.button_station_save.setText("保存(S)")
        self.button_station_save.move(440, 20)
        self.button_station_save.setFixedHeight(30)
        self.button_station_save.setFixedWidth(120)
        self.button_station_save.setStyleSheet("font:14px Microsoft YaHei; background-color: rgb(225, 225, 225); "
                                               "border: 1px solid rgb(173, 173, 173); border-radius: 3px")
        self.button_station_save.clicked.connect(lambda: self.save(self))

        self.tableview = QTableView(self)
        self.tableview.move(20, 70)
        self.tableview.setFixedHeight(480)
        self.tableview.setFixedWidth(760)
        self.tableview.setStyleSheet("color:black; background-color:white; border-radius: 3px")
        self.tableview.horizontalHeader().setStyleSheet(
            "color:black; background-color:rgb(47, 52, 73); border-radius: 3px")
        self.tableview.verticalHeader().setStyleSheet(
            "color:black; background-color:rgb(47, 52, 73); border-radius: 3px")
        self.tableview.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableview.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableview.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableview.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableview.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tableview.setShowGrid(True)
        self.tableview.setAlternatingRowColors(True)
        self.tableview.setSortingEnabled(True)
        self.tableview.setMouseTracking(True)

        self.tableview.setContextMenuPolicy(Qt.CustomContextMenu)
        self.context_menu = QMenu()
        self.tableview.customContextMenuRequested.connect(self.show_context_menu)

        # 初始化电站信息
        self.tableview_model = QStandardItemModel()
        self.tableview_model.setHorizontalHeaderLabels(["电站名称", "电站类型", "装机容量", "备注"])
        self.tableview_model.setRowCount(0)
        self.tableview_model.setColumnCount(4)
        try:
            with open(base_path + f"/data/{province}_station.csv", "r") as self.station_file:
                self.station_file_reader = csv.reader(self.station_file)
                station_list = [row for row in self.station_file_reader]
                for i in range(len(station_list)):
                    for j in range(4):
                        try:
                            self.tableview_model.setItem(i, j, QStandardItem(station_list[i][j]))
                        except IndexError:
                            pass
            self.station_file.close()
        except FileNotFoundError:
            pass

        self.tableview.setModel(self.tableview_model)

    def change_province(self, index):
        try:
            exec(f"self.station_window{index}.show()")
            self.hide()
        except AttributeError:
            self.hide()
            exec(f"self.station_window{index} = StationWindow(province_list[{index}])")
            exec(f"self.station_window{index}.show()")

    class new_window(QWidget):

        def __init__(self, station_window):
            super().__init__()

            self.setWindowTitle("新建电站")
            self.setFixedSize(400, 300)
            self.setWindowIcon(QIcon(base_path + "/images/软件图标.png"))

            self.label_station_name = QLabel(self)
            self.label_station_name.setText("电站名称：")
            self.label_station_name.move(20, 20)
            self.label_station_name.setFixedWidth(80)
            self.label_station_name.setFixedHeight(30)
            self.label_station_name.setStyleSheet("font:14px Microsoft YaHei;")

            self.input_station_name = QLineEdit(self)
            self.input_station_name.move(100, 20)
            self.input_station_name.setFixedWidth(280)
            self.input_station_name.setFixedHeight(30)
            self.input_station_name.setStyleSheet("font:14px Microsoft YaHei;")

            self.label_station_type = QLabel(self)
            self.label_station_type.setText("电站类型：")
            self.label_station_type.move(20, 70)
            self.label_station_type.setFixedWidth(80)
            self.label_station_type.setFixedHeight(30)
            self.label_station_type.setStyleSheet("font:14px Microsoft YaHei;")

            self.input_station_type = QComboBox(self)
            self.input_station_type.move(100, 70)
            self.input_station_type.setFixedWidth(280)
            self.input_station_type.setFixedHeight(30)
            self.input_station_type.setStyleSheet("font:14px Microsoft YaHei;")
            self.input_station_type.addItems(["电化学", "抽水蓄能", "氢储能"])

            self.label_station_capacity = QLabel(self)
            self.label_station_capacity.setText("装机容量：")
            self.label_station_capacity.move(20, 120)
            self.label_station_capacity.setFixedWidth(80)
            self.label_station_capacity.setFixedHeight(30)
            self.label_station_capacity.setStyleSheet("font:14px Microsoft YaHei;")

            self.input_station_capacity = QLineEdit(self)
            self.input_station_capacity.move(100, 120)
            self.input_station_capacity.setFixedWidth(280)
            self.input_station_capacity.setFixedHeight(30)
            self.input_station_capacity.setStyleSheet("font:14px Microsoft YaHei;")

            self.label_station_remark = QLabel(self)
            self.label_station_remark.setText("备注：")
            self.label_station_remark.move(20, 170)
            self.label_station_remark.setFixedWidth(80)
            self.label_station_remark.setFixedHeight(30)
            self.label_station_remark.setStyleSheet("font:14px Microsoft YaHei;")

            self.input_station_remark = QLineEdit(self)
            self.input_station_remark.move(100, 170)
            self.input_station_remark.setFixedWidth(280)
            self.input_station_remark.setFixedHeight(30)
            self.input_station_remark.setStyleSheet("font:14px Microsoft YaHei;")

            self.button_confirm = QPushButton("确定", self)
            self.button_confirm.move(100, 220)
            self.button_confirm.setFixedWidth(80)
            self.button_confirm.setFixedHeight(30)
            self.button_confirm.setStyleSheet("font:14px Microsoft YaHei;")

            self.button_cancel = QPushButton("取消", self)
            self.button_cancel.move(300, 220)
            self.button_cancel.setFixedWidth(80)
            self.button_cancel.setFixedHeight(30)
            self.button_cancel.setStyleSheet("font:14px Microsoft YaHei;")

            self.button_confirm.clicked.connect(lambda: self.confirm(station_window))
            self.button_cancel.clicked.connect(self.cancel)

        def confirm(self, station_window):
            station_window.tableview_model.appendRow([
                QStandardItem(self.input_station_name.text()),
                QStandardItem(self.input_station_type.currentText()),
                QStandardItem(self.input_station_capacity.text()),
                QStandardItem(self.input_station_remark.text())
            ])
            self.close()

        def cancel(self):
            self.close()

    def show_context_menu(self, pos):
        self.context_menu.clear()
        edit_action = QAction('编辑', self)
        delete_action = QAction('删除', self)
        self.context_menu.addAction(edit_action)
        self.context_menu.addAction(delete_action)
        action = self.context_menu.exec_(self.tableview.mapToGlobal(pos))

        if action == edit_action:
            row = self.tableview.currentIndex().row()
            self.edit_list = []
            for col in range(self.tableview.model().columnCount()):
                editor = QLineEdit(self.tableview)
                editor.setText(self.tableview.model().data(self.tableview.model().index(row, col)))
                editor.installEventFilter(self)
                self.tableview.setIndexWidget(self.tableview.model().index(row, col), editor)
                self.edit_list.append(editor)

        elif action == delete_action:
            self.tableview_model.removeRow(self.tableview.currentIndex().row())

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Return:
            row = self.tableview.currentIndex().row()
            for col in range(self.tableview.model().columnCount()):
                self.tableview_model.setData(self.tableview_model.index(row, col), self.edit_list[col].text())
                self.tableview.setIndexWidget(self.tableview.model().index(row, col), None)
            return True
        return super().eventFilter(obj, event)

    def save(self, station_window):
        # rearrange the model，氢储能在前，电化学中间，抽水蓄能在后
        for row in range(self.tableview_model.rowCount()):
            if self.tableview_model.data(self.tableview_model.index(row, 1)) == "氢储能":
                self.tableview_model.insertRow(0, self.tableview_model.takeRow(row))
            elif self.tableview_model.data(self.tableview_model.index(row, 1)) == "抽水蓄能":
                self.tableview_model.insertRow(self.tableview_model.rowCount() - 1, self.tableview_model.takeRow(row))
        with open(f"data/{station_window.province}_station.csv", "w") as station_file:
            station_file.truncate(0)
            writer = csv.writer(station_file, delimiter=',', lineterminator='\n')
            for row in range(self.tableview_model.rowCount()):
                for col in range(self.tableview_model.columnCount()):
                    if self.tableview_model.data(self.tableview_model.index(row, col)) == "":
                        self.tableview_model.setData(self.tableview_model.index(row, col), " ")
                writer.writerow([
                    self.tableview_model.data(self.tableview_model.index(row, 0)),
                    self.tableview_model.data(self.tableview_model.index(row, 1)),
                    self.tableview_model.data(self.tableview_model.index(row, 2)),
                    self.tableview_model.data(self.tableview_model.index(row, 3))
                ])


class ResultWindow(QWidget):

    def __init__(self, entry_window: EntryWindow, method: str, num):
        super().__init__()
        self.entry_window = entry_window
        self.method = method

        self.setWindowTitle(str(province_list[num]))
        self.setFixedSize(1280, 720)

        self.setWindowIcon(QIcon(base_path + "/images/软件图标.png"))

        self.background_image = QLabel(self)
        self.background_image.setPixmap(QPixmap(base_path + f"/images/运行结果（{province_list[num]}）.png"))

        # 设置导出按钮
        self.button_export = QPushButton("导出", self)
        self.button_export.move(20, 30)
        self.button_export.setFixedHeight(30)
        self.button_export.setFixedWidth(90)
        self.button_export.setStyleSheet("color:white; background-color:rgb(47, 52, 73); border-radius: 3px")
        self.button_export.clicked.connect(self.save_file)

        self.button_ai = QPushButton(self)
        self.button_ai.setText("返回")
        self.button_ai.move(120, 30)
        self.button_ai.setFixedHeight(30)
        self.button_ai.setFixedWidth(90)
        self.button_ai.setStyleSheet("color:white; background-color:rgb(47, 52, 73); border-radius: 3px")
        self.button_ai.clicked.connect(lambda: entry_window.show())
        self.button_ai.clicked.connect(lambda: self.hide())

        # 设置省份按钮
        for i in province_list:
            exec(f"self.button{province_list.index(i)}_province = QPushButton(self)")
            exec(f"self.button{province_list.index(i)}_province.setText(i)")
            # 确定按钮位置，在(20,  130 + 30 * index)处
            exec(f"self.button{province_list.index(i)}_province.move(20, {130 + 30 * province_list.index(i)})")
            exec(f"self.button{province_list.index(i)}_province.setFixedHeight(28)")
            exec(f"self.button{province_list.index(i)}_province.setFixedWidth(90)")
            exec(f"self.button{province_list.index(i)}_province.setStyleSheet(\"color:white; background-color:rgb(47, "
                 f"52, 73); border-radius: 3px\")")
            exec(f"self.button{province_list.index(i)}_province.clicked.connect(partial(entry_window.calculate, "
                 f"{province_list.index(i)}))")
            exec(f"self.button{province_list.index(i)}_province.clicked.connect(self.hide)")

        # 设置时间轴
        self.time_slider = QSlider(Qt.Horizontal, self)
        self.time_slider.setFixedWidth(180)
        self.time_slider.setRange(0, 95)
        self.time_slider.setSingleStep(1)
        self.time_slider.setStyleSheet("QSlider::handle:horizontal{background-color:#3f4459; \
                                      width: 15px; margin: -7px -7px -7px -7px; border-radius: 3px;}")

        self.time_slider.move(140, 607)
        # 链接时间轴与时间对象label
        self.time_slider.valueChanged.connect(self.time_change)
        # 链接时间轴与储能对象出力label
        self.time_slider.valueChanged.connect(self.power_change)

        # 创建时间对象label
        self.time_label = QLabel(self)
        self.time_label.setText("--:--")
        self.time_label.setStyleSheet("font:24px Microsoft YaHe; font-weight:700; color:rgb(47, 52, 73)")
        self.time_label.setFixedWidth(120)
        self.time_label.move(330, 605)

        # 创建储能对象出力label
        self.power_label_list = []
        for i in range(len(self.entry_window.H2_power_list)):
            self.power_label_list.append(QLabel(self))
            string = f'{i + 1}'.ljust(2) + '   ' + '氢储能' + '    ' + '--------'
            self.power_label_list[i].setText(string)
            self.power_label_list[i].setStyleSheet("font:18px Microsoft YaHe; font-weight:700; color:rgb(47, 52, 73)")
            self.power_label_list[i].move(180, 164 + 24 * i)
        for i in range(len(self.entry_window.ele_power_list)):
            self.power_label_list.append(QLabel(self))
            string = f'{i + len(self.entry_window.H2_power_list) + 1}'.ljust(2) + '   ' + '电化学' + '    ' + '--------'
            self.power_label_list[i + len(self.entry_window.H2_power_list)].setText(string)
            self.power_label_list[i + len(self.entry_window.H2_power_list)].setStyleSheet(
                "font:18px Microsoft YaHe; font-weight:700; color:rgb(47, 52, 73)")
            self.power_label_list[i + len(self.entry_window.H2_power_list)]. \
                move(180, 164 + 24 * (i + len(self.entry_window.H2_power_list)))
        for i in range(len(self.entry_window.water_power_list)):
            self.power_label_list.append(QLabel(self))
            string = f'{i + len(self.entry_window.H2_power_list) + len(self.entry_window.ele_power_list) + 1}'.ljust(2) \
                     + '  ' + '抽水蓄能' + '   ' + '--------'
            self.power_label_list[i + len(self.entry_window.H2_power_list) + len(self.entry_window.ele_power_list)]. \
                setText(string)
            self.power_label_list[i + len(self.entry_window.H2_power_list) + len(self.entry_window.ele_power_list)]. \
                setStyleSheet("font:18px Microsoft YaHe; font-weight:700; color:rgb(47, 52, 73)")
            self.power_label_list[i + len(self.entry_window.H2_power_list) + len(self.entry_window.ele_power_list)]. \
                move(180, 164 + 24 * (i + len(self.entry_window.H2_power_list) + len(self.entry_window.ele_power_list)))

        # 创建cost label
        self.cost_label = QLabel(self)
        self.cost_label.setText(f"{str(self.entry_window.cost[0][0])[0:9]}元/天")
        self.cost_label.setStyleSheet("font:22px Microsoft YaHe; font-weight:700; color:rgb(47, 52, 73)")
        self.cost_label.move(240, 660)

        self.form_window = QWidget(self)
        self.form_window.move(890, 80)
        self.form_window.resize(350, 600)
        self.form_window.setStyleSheet("background-color:rgb(195, 198, 197)")
        self.form_window.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.form = QFormLayout(self.form_window)
        self.form.setVerticalSpacing(10)

        # 设置一个遮罩层，用于显示三种储能方式分别的叠加出力
        self.mask_window = QWidget(self)
        self.mask_window.move(890, 80)
        self.mask_window.resize(350, 600)
        self.mask_window.setStyleSheet("background-color:rgb(195, 198, 197)")
        storage_method_list = ['氢储能', '电化学', '抽水蓄能']
        for i in storage_method_list:
            x = [j for j in range(96)]
            y = [0 for j in range(96)]
            for j in range(96):
                if i == '氢储能':
                    y[j] = sum(
                        self.entry_window.H2_power_list[k][j] for k in range(len(self.entry_window.H2_power_list)))
                elif i == '电化学':
                    y[j] = sum(
                        self.entry_window.ele_power_list[k][j] for k in range(len(self.entry_window.ele_power_list)))
                else:
                    y[j] = sum(self.entry_window.water_power_list[k][j] for k in
                               range(len(self.entry_window.water_power_list)))
            fig = plt.figure(figsize=(350 / 100, 200 / 100), dpi=100)
            ax = fig.add_subplot(111)
            plt.plot(x, y)
            plt.tick_params(labelsize=6)
            plt.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.89)
            plt.title(f"{i}出力图", font="Microsoft YaHei", fontsize=10)
            plt.xlabel("时间/15min", font="Microsoft YaHei", fontsize=8)
            plt.ylabel("出力/kW", font="Microsoft YaHei", fontsize=8, labelpad=-2)
            figure_canvas = FigureCanvas(fig)
            buffer = io.BytesIO()
            figure_canvas.print_png(buffer)
            buffer.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue())
            plt.close(fig)
            label = QLabel(self.mask_window)
            label.setPixmap(pixmap)
            label.move(QPoint(0, storage_method_list.index(i) * 200))
            label.resize(350, 200)
        self.mask_window.show()
        self.switch_button0 = QPushButton(self.mask_window)
        self.switch_button0.setIcon(QIcon("images/切换.jpg"))
        self.switch_button0.setIconSize(QSize(20, 20))
        self.switch_button0.move(0, 0)
        self.switch_button0.clicked.connect(lambda: self.image_show())

    # 设置导出文件函数
    def save_file(self):
        saving_file, _ = QFileDialog.getSaveFileName(self, "保存文件", "D://", "CSV Files (*.csv)")
        if saving_file:
            with open(saving_file, "w") as file:
                file.truncate(0)
                writer = csv.writer(file, delimiter=',', lineterminator='\n')
                for row in range(self.entry_window.power_list.__len__()):
                    writer.writerow(self.entry_window.power_list[row])
                writer.writerow(self.entry_window.cost)

    # 设置时间轴与时间对象label的链接函数
    def time_change(self, new_value):
        if new_value // 4 < 10:
            if new_value % 4 == 0:
                self.time_label.setText("0{0}:0{1}".format(new_value // 4, 15 * (new_value % 4)))
            else:
                self.time_label.setText("0{0}:{1}".format(new_value // 4, 15 * (new_value % 4)))
        else:
            if new_value % 4 == 0:
                self.time_label.setText("{0}:0{1}".format(new_value // 4, 15 * (new_value % 4)))
            else:
                self.time_label.setText("{0}:{1}".format(new_value // 4, 15 * (new_value % 4)))

    # 设置时间轴与储能对象出力label的链接函数
    def power_change(self, new_value):
        for i in range(len(self.entry_window.power_list)):
            string = self.power_label_list[i].text()
            string0 = str(self.entry_window.power_list[i][new_value]).ljust(8)
            string0 = string0[:8]
            string = string[:-8] + string0
            self.power_label_list[i].setText(string)

    # 设置储能对象出力图的链接函数
    def image_show(self):
        self.mask_window.hide()
        switch_button = QPushButton(self.form_window)
        switch_button.setIcon(QIcon("images/切换.jpg"))
        switch_button.setIconSize(QSize(20, 20))
        self.form.addRow(switch_button)
        switch_button.clicked.connect(lambda: self.mask_window.show())
        switch_button.clicked.connect(lambda: self.mask_window.raise_())
        self.switch_button0.disconnect()
        self.switch_button0.clicked.connect(lambda: self.mask_window.hide())

        for i in range(len(self.entry_window.power_list)):
            x = [j for j in range(96)]
            y = []
            for j in range(96):
                y.append(self.entry_window.power_list[i][j])
            fig = plt.figure(figsize=(300 / 100, 400 / 100), dpi=100)
            ax = fig.add_subplot(111)
            plt.plot(x, y)
            plt.tick_params(labelsize=6)
            plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.89)
            plt.title(f"{i + 1}号储能对象出力图", font="Microsoft YaHei", fontsize=10)
            plt.xlabel("时间/15min", font="Microsoft YaHei", fontsize=8)
            plt.ylabel("出力/kW", font="Microsoft YaHei", fontsize=8, labelpad=-2)
            figure_canvas = FigureCanvas(fig)

            buffer = io.BytesIO()
            figure_canvas.print_png(buffer)
            buffer.seek(0)

            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue())
            plt.close(fig)

            label = QLabel()
            label.setPixmap(pixmap)
            self.form.addRow(label)

            exec(f"Button{i + 1} = QPushButton()")
            exec(f"Button{i + 1}.setFixedSize(20, 20)")
            exec(f"Button{i + 1}.setIcon(QIcon('images/放大.jpg'))")
            exec(f"Button{i + 1}.setIconSize(QSize(20, 20))")
            # 按下按钮后弹出新窗口
            exec(f"Button{i + 1}.clicked.connect(partial(self.enlarge, {i + 1}))")

            exec(f"self.form.addRow(Button{i + 1})")

        scroll = QScrollArea(self)
        scroll.setWidget(self.form_window)
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.move(890, 80)
        scroll.resize(350, 600)
        scroll.setStyleSheet("background-color:rgb(195, 198, 197)")
        scroll.show()

    def enlarge(self, number):
        try:
            exec(f"self.enlarge_window{number}.show()")
        except AttributeError:
            exec(f"self.enlarge_window{number} = self.enlarge_window(self, number)")
            exec(f"self.enlarge_window{number}.show()")

    class enlarge_window(QWidget):

        def __init__(self, result_window, number):
            super().__init__()

            self.setWindowTitle(f"{number}号储能对象出力图")
            self.resize(800, 600)
            self.setStyleSheet("background-color:rgb(255, 255, 255)")

            x = [j for j in range(96)]
            y = []
            for j in range(96):
                y.append(result_window.entry_window.power_list[number - 1][j])
            self.fig = plt.figure(figsize=(800 / 100, 600 / 100), dpi=100)
            ax = self.fig.add_subplot(111)
            plt.plot(x, y)
            plt.tick_params(labelsize=10)
            plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9)
            plt.title(f"{number}号储能对象出力图", font="Microsoft YaHei", fontsize=16)
            plt.xlabel("时间/15min", font="Microsoft YaHei", fontsize=14)
            plt.ylabel("出力/kW", font="Microsoft YaHei", fontsize=14, labelpad=-2)
            self.figure_canvas = FigureCanvas(self.fig)

            self.buffer = io.BytesIO()
            self.figure_canvas.print_png(self.buffer)
            self.buffer.seek(0)

            self.pixmap = QPixmap()
            self.pixmap.loadFromData(self.buffer.getvalue())
            plt.close(self.fig)

            self.label = QLabel(self)
            self.label.setPixmap(self.pixmap)
            self.label.move(0, 0)
            self.label.resize(800, 600)

            self.show()

    # 弹出退出提示框
    def closeEvent(self, event):
        reply = QMessageBox.question(self, '提示', '是否要退出程序？', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


# 主程序入口
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EntryWindow()
    window.show()
    sys.exit(app.exec_())
