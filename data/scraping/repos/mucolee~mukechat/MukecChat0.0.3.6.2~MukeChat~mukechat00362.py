# -*- coding: utf-8 -*-
import json
import sys
import time

import openai

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QDesktopWidget
from openai.error import APIConnectionError, AuthenticationError

# from MukeChat import Setting2306021  # 打包时
import Setting2306021
chat_record_file_path = 'mukechat_file/chat_record.txt'
setting_data_path = 'mukechat_file/setting_date.txt'
conversation = []  # 聊天记忆
chat_record = []  # 聊天记录
with open(chat_record_file_path, 'r') as f:
    chat_record = json.load(f)
# with open(setting_data_path, 'r') as f:
#     setting_date = json.load(f)
#     #print(setting_date)
#     print(setting_date['api_key'])
#     print(Setting2306021.setting_date['api_key'])
icon_path = 'mukechat_file/logo.ico'


# 按钮的多线程
class SubmitThread(QThread):
    ai_re_signal = pyqtSignal(str)

    def __init__(self, target=None):  # 23.05.22 07:55
        super().__init__()
        self.ai_re_text = None
        self.prompt = ''
        # self._target = target

    def ai_reply(self) -> str:
        try:
            prompt = self.prompt
            openai.api_key = Setting2306021.setting_date['api_key']
            global conversation, chat_record
            max_history_len = 10  # 保存5个会话给chatgpt  AI记忆控制

            conversation.append({"role": "user", "content": prompt})
            chat_record.append({"role": "user", "content": prompt})

            if len(conversation) > max_history_len:
                conversation = chat_record[-max_history_len:]  # 删掉数组第一个元素

            response = openai.ChatCompletion.create(  # 调用官方给出来的API接口获取返回值
                model="gpt-3.5-turbo",
                messages=conversation,
                temperature=Setting2306021.setting_date['temperature'],
                max_tokens=Setting2306021.setting_date['max_tokens'],
                n=1,
                timeout=15,
                stop=None,
                top_p=Setting2306021.setting_date['top_p']
            )
            # 把ai的回复消息放到列表里
            conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
            chat_record.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
            return response['choices'][0]['message']['content']
        except APIConnectionError:
            return '请求失败！请检查网络是否畅通！大陆地区需要科学上网！你可以通过邮箱向mukelee0723@Gmail.com请求帮助！'
        except KeyboardInterrupt:
            return '\n连接已经被用户手动断开'
        except AuthenticationError:
            return '请检查你的密钥是否有问题！请聚焦主窗口并通过ait+k快捷键打开密钥设置！'
        except Exception:
            return '未知错误！前联系开发人员！邮箱:mukelee0723@Gmail.com！对被采纳的反馈将会得到相应报酬！'

    def get_user_prompt(self, prompt):  # 获取用户请求内容
        self.prompt = prompt

    def run(self) -> None:
        prompt = self.ai_reply()  # 调用接口
        print(prompt)
        temp = ' AI:\n     '
        for item in prompt:
            temp += item
            self.ai_re_signal.emit(temp)  # 把Ai回复以信号的方式传回去
            time.sleep(0.03)  # 让界面出现逐字出现的的动画效果
        # self._target()


# 用QLabel重写聊天气泡
class TextBox(QLabel):
    def __init__(self, parent: QWidget, layout: object):
        super().__init__()
        self.setParent(parent)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.setFont(font)
        self.setWordWrap(True)
        self.setMargin(20)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # self.setReadOnly(True)
        self.setPixmap(QPixmap('MukeChat230607/马.png'))
        self.setStyleSheet('border-width: 1px;border-style: solid;'
                           'border-color: rgb(255, 170, 0);'
                           'background-color: rgb(100, 149, 237);'
                           'border-radius: 20px;')
        layout.addWidget(self)


# 主窗口UI
class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(900, 800)
        # Form.setWindowOpacity(0.9)
        # Form.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        # Form.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        Form.setMinimumSize(QtCore.QSize(400, 350))
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget = QtWidgets.QWidget(Form)
        self.widget.setObjectName("widget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.scrollArea = QtWidgets.QScrollArea(self.widget)
        # self.scrollArea.setStyleSheet('QScrollBar{background-color: white; border: 1px solid black;}')

        self.scrollArea.verticalScrollBar().rangeChanged.connect(  # 滚动区发生变化时，滚到滚动区最大值
            lambda: self.scrollArea.verticalScrollBar().setValue(
                self.scrollArea.verticalScrollBar().maximum()
            )
        )
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 408, 280))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        # 水平布局
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.verticalLayout_2.addStretch(1)  # 类似弹簧

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.horizontalLayout_2.addWidget(self.scrollArea)
        self.verticalLayout.addWidget(self.widget)
        self.widget_2 = QtWidgets.QWidget(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_2.sizePolicy().hasHeightForWidth())
        self.widget_2.setSizePolicy(sizePolicy)
        self.widget_2.setMinimumSize(QtCore.QSize(0, 105))
        self.widget_2.setMaximumSize(QtCore.QSize(16777215, 130))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.widget_2.setFont(font)
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout.setObjectName("horizontalLayout")
        mukechat_introduce = TextBox(parent=self.scrollAreaWidgetContents,
                                     layout=self.verticalLayout_2)
        mukechat_introduce.setText("你好!我是由Muke开发，基于c/s架构的Chat-GPT客户端程序MukeChat！通过我你可以间接访问到OpenAI提供的"
                                   "Chat-GPT接口，当然这前提是你有GPT的密钥和在电脑上搭建虚拟专用网络（VPN，因为GPT接口并不对中国大陆地区开放）。"
                                   "你可以通过邮箱mukelee0723@gamil.com联系Muke以获取支持或通过下面二维码给Muke买一杯奶茶！")

        QR_code = TextBox(parent=self.scrollAreaWidgetContents,
                          layout=self.verticalLayout_2)
        QR_code.setPixmap(QPixmap('mukechat_file/QR_code.png'))
        QR_code.setAlignment(Qt.AlignCenter)
        # QR_code.setScaledContents (True)

        # 输入框
        self.textEdit_input = myTextEdit(self.widget_2, self)
        self.textEdit_input.setPlaceholderText('请输入……')
        self.textEdit_input.setMinimumSize(QtCore.QSize(0, 86))
        self.textEdit_input.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.IBeamCursor))
        self.textEdit_input.setObjectName("textEdit_input")
        self.horizontalLayout.addWidget(self.textEdit_input)

        # 提交按钮
        self.pushButton = QtWidgets.QPushButton(self.widget_2)
        self.pushButton.clicked.connect(self.start_btn_thread)  # 绑定事件
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setMinimumSize(QtCore.QSize(70, 86))
        self.pushButton.setFont(font)
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.verticalLayout.addWidget(self.widget_2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    # 添加聊天框，开启请求线程
    def start_btn_thread(self):
        user_text = self.textEdit_input.toPlainText().strip()
        if user_text != '':
            # 创建文字聊天气泡
            chat_box_user = TextBox(parent=self.scrollAreaWidgetContents,
                                    layout=self.verticalLayout_2)
            chat_box_user.setStyleSheet('border-width: 1px;border-style: solid;'
                                        'border-color: rgb(255, 170, 0);'
                                        'background-color: rgb(200, 179, 201);'
                                        'border-radius: 20px;')
            # chat_box_user.setStyleSheet('background:pink;')
            chat_box_user.setText('User:\n      ' + user_text + '\n')
            self.textEdit_input.setText('')  # 清空
            self.chat_box_ai = TextBox(parent=self.scrollAreaWidgetContents,
                                       layout=self.verticalLayout_2)
            self.chat_box_ai.setStyleSheet('border-width: 1px;border-style: solid;'
                                           'border-color: rgb(255, 170, 0);'
                                           'background-color: rgb(252, 246, 207);'
                                           'border-radius: 20px;')

            # 创建线程
            self.thread = SubmitThread()
            self.thread.get_user_prompt(user_text)
            self.thread.ai_re_signal.connect(lambda text: self.chat_box_ai.setText(text + '\n'))  # 绑定信号的槽函数
            # print(self.thread)

            self.thread.start()
            # print(self.text)

    def get_ai_re(self, text):  # 信号的槽函数
        self.chat_box_ai.setText(text)  # 把子线程发过来的东西在主线UI上打印出来（通过线程通讯处理），
        # 子线程上操作主线程UI会出现闪退现象，这是由于Qt存在兼容问题

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "MukeChat"))
        Form.setWindowIcon(QIcon(icon_path))
        self.pushButton.setText(_translate("Form", "发送"))


# mukechat主窗口UI在封装对象
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # 窗体居中
        center_pointer = QDesktopWidget().availableGeometry().center()
        x = center_pointer.x()
        y = center_pointer.y()
        win_w = 900
        win_h = 800
        self.move(x - win_w // 2, y - win_h // 2)

        self.win_ui = Ui_Form()
        self.win_ui.setupUi(self)
        # 初始化数据
        self.init_record_date()
        # self.win_ui.scrollAreaWidgetContents

    # 初始化聊天记录，把之前的聊天记录加载上来
    def init_record_date(self):
        for item in chat_record:
            if item['role'] == 'user':
                self.win_ui.user_record = TextBox(parent=self.win_ui.scrollAreaWidgetContents,
                                                  layout=self.win_ui.verticalLayout_2)
                self.win_ui.user_record.setStyleSheet('border-width: 1px;border-style: solid;'
                                                      'border-color: rgb(176, 194, 178);'
                                                      'background-color: rgb(200, 179, 201);'
                                                      'border-radius: 20px;')
                self.win_ui.user_record.setText(' User:\n      ' + item['content'] + '\n')
            else:
                self.win_ui.AI_record = TextBox(parent=self.win_ui.scrollAreaWidgetContents,
                                                layout=self.win_ui.verticalLayout_2)
                self.win_ui.AI_record.setStyleSheet('border-width: 1px;border-style: solid;'
                                                    'border-color: rgb(176, 194, 178);'
                                                    'background-color: rgb(252, 246, 207);'
                                                    'border-radius: 20px;')
                self.win_ui.AI_record.setText(' AI:\n     ' + item['content'] + '\n')

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        pass

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:  # 主窗口监听
        # alt+k 密钥快捷设置
        if a0.modifiers() == QtCore.Qt.KeyboardModifier.AltModifier and a0.key() == QtCore.Qt.Key_K:
            # self.mydialog = MyDialog(self)  # 子窗口
            dialog = Setting2306021.MyDialog()  # Setting子窗口
            dialog.exec_()

        if a0.key() == QtCore.Qt.Key_A:
            pass

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:  # 重写窗口关闭事件，关闭前保存聊天数据
        with open(chat_record_file_path, 'w') as file:
            json.dump(chat_record, file)


# 继承了keyPressEvent的QTextEdit，可以进行键盘监听
class myTextEdit(QtWidgets.QTextEdit):
    def __init__(self, parent, current_class_obj=None):
        QtWidgets.QTextEdit.__init__(self)
        self.parent = parent
        self.current_class_obj = current_class_obj
        self.toPlainText()

    def keyPressEvent(self, event):  # 文本框监听
        QtWidgets.QTextEdit.keyPressEvent(self, event)  # 继承监听事件
        if event.key() == Qt.Key_Return:
            if self.toPlainText().strip() != '':
                self.current_class_obj.start_btn_thread()


if __name__ == '__main__':
    # read_file()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
