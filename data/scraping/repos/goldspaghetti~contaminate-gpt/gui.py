import typing
from PyQt6 import QtGui
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel, QProgressBar, QPushButton, QSizePolicy
from PyQt6.QtGui import QIcon, QPainter, QColor, QPixmap, QPen, QBrush, QLinearGradient, QImage, QPainterPath, QBitmap
# from PyQt6.QtCore import Qt, QSize, QRect, QPointF, QRectF, QEvent, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtCore import *
# from PySide6.QtCore import Signal, Slot, QObject, QThread

from functools import partial
from random import randint
import time
import openai

import threading
import os

def clamp(n, l, h): 
    return max(l, min(n, h))

class Timer(QLabel):
    def __init__(self):
        super().__init__("0:00")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.min = 3
        self.sec = 0

        self.setStyleSheet("""
            QWidget {
                background: rgb(20, 20, 20);
                border-radius: 10px;
                font-size: 18px;
                font-weight: 700;
                padding: 8px;
            }
        """)

    def __update_time(self):
        self.setText(f"{self.min:02}:{self.sec:02}")

    def set_time(self, min, sec):
        self.min = min
        self.sec = sec

        self.__update_time()

    def decrement_time(self):
        if self.sec == 0:
            self.min -= 1
            self.sec = 59
        else:
            self.sec -= 1

        self.__update_time()

class Health(QProgressBar):
    def __init__(self):
        super().__init__()
        self.setValue(100)
#                 background: rgb(19, 70, 158);
#                 background: rgb(145, 25, 12);
        self.setStyleSheet("""
            QProgressBar {
                font-weight: 700;
                background: rgb(20, 20, 20);
                border-radius: 10px;
                text-align: center;
                margin: 4px;
                height: 30px;
            }
                           
            QProgressBar::chunk {
                background: rgb(19, 70, 158);
                border-radius: 10px;
            }
        """)

    def decrease_health(self, percentage):
        self.setValue(self.value() - percentage)

class TopBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self.timer = Timer()
        self.health = Health()

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.timer)
        self.layout.addStretch()
        self.layout.addWidget(self.health)
        # self.layout.addStretch()

        self.setLayout(self.layout)

        self.setFixedHeight(50)
        #                 background: rgb(30, 84, 176);
        self.setStyleSheet("""
            QWidget {
                background: rgb(30, 30, 30);
            }
        """)

class Item(QPushButton):
    def __init__(self, path, damage):
        super().__init__()
        self.path = path
        self.damage = damage

        self.setFixedSize(100,100)

        self.setIcon(QIcon(path))
        self.setIconSize(QSize(100, 100))
#                 background: rgb(30,30,30);
#                 background: rgb(8, 57, 140);
#                 background: rgb(19, 70, 158);
        self.setStyleSheet("""
            QPushButton {
                background: rgb(20, 20, 20);
                border-radius: 10px;
            }
                           
            QPushButton::hover {
                background: rgb(30, 30, 30);
                border-radius: 10px;
            }
        """)

        # print("hi")
        # self.setSizePolicy(self.sizePolicy().setRetainSizeWhenHidden(True))

# class Window(QScrollArea):
#     def __init__(self):
#         super(Window, self).__init__()
#         widget = QWidget()
#         layout = QVBoxLayout(widget)
#         layout.setAlignment(Qt.AlignTop)
#         for index in range(100):
#             layout.addWidget(QLabel('Label %02d' % index))
#         self.setWidget(widget)
#         self.setWidgetResizable(True)

class SideBar(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # self.setStyleSheet("background: red")
        self.parent = parent
        self.layout = QVBoxLayout()
        # self.layout.setAlignment(Qt.AlignmentFlag)

        for path in os.listdir('images'):
            print(path)
            item = Item(f"images/{path}", randint(10,30))
            self.layout.addWidget(item)

            item.clicked.connect(partial(self.item_event, item))
        # self.layout.addWidget(Box())
        # self.layout.addWidget(Box())
        # self.layout.addWidget(Box())
        # self.layout.addWidget(Box())
        # self.layout.addWidget(Box())
        # self.layout.addWidget(Box())
        # self.layout.addWidget(Box())
        # self.layout.addWidget(Box())
        self.layout.addStretch()
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(20,20,20,20)
        self.setLayout(self.layout)
        # ok = Box()
        # ok.show()
        # ok.move(100,100)

        # self.setFixedHeight(140)
        self.setFixedWidth(140)
        # self.setStyleSheet("""
        #     QWidget {
        #         background: rgb(20,20,20);
        #     }
        # """)

    def item_event(self, item):
        if self.parent.active_path:
            return

        self.parent.active_path = item.path
        self.parent.active_damage = item.damage
        item.setVisible(False)

        self.parent.repaint()

        self.parent.targets = [
            (randint(260, 900), randint(240, 500)),
            (randint(260, 900), randint(240, 500)),
            (randint(260, 900), randint(240, 500))
        ]


class Lip():
    def __init__(self, event_queue=None):
        super().__init__()
        self.event_queue = event_queue
        self.curr_event = None
        self.last_time = 0
        lip_thread = threading.Thread(target=self.thread_loop, daemon=True)
        lip_thread.start()
        self.last_face = "N"
        self.img_path = "lip_X.png"

    
    def thread_loop(self):
        while True:
            self.check_event()
            self.update_loop()
            time.sleep(0.05)

    def update_loop(self):
        if self.curr_event != None:
            curr_face = "N"
            for event in self.curr_event:
                if time.time() - self.last_time < float(event[0]):
                    curr_face = event[1]
                    break
            if curr_face == "N":
                self.curr_event = None
                return
            if curr_face != self.last_face:
                print(f"change image to {curr_face}, last face {self.last_face}, {curr_face == self.last_face} time: {time.time()-self.last_time}")
                self.img_path = f"lip_{curr_face}.png"
                self.last_face = curr_face

    def check_event(self):
        if not self.event_queue.empty():
            print("queue not empty")
            self.curr_event = self.event_queue.get()
            self.last_time = time.time()
            self.last_face = "N"

class MainWindow(QMainWindow):
    def __init__(self, lip_queue=None, tts=None):
        super().__init__()
        self.setMouseTracking(True)
        self.setFixedSize(1000, 600)

        self.setStyleSheet("""
            QMainWindow {
                background: white;
            }
        """)

        self.game_over = False
        self.shield = None
        self.active_path = None
        self.active_damage = 0
        self.targets = []
        self.splats = []

        self.x = 1000
        self.y = 1000

        self.top_bar = TopBar()
        self.side_bar = SideBar(self)

        self.bottom_half = QWidget()
        self.bottom_half.layout = QHBoxLayout()
        self.bottom_half.layout.addWidget(self.side_bar)
        self.bottom_half.layout.addStretch()
        self.bottom_half.layout.setContentsMargins(0,0,0,0)
        self.bottom_half.setLayout(self.bottom_half.layout)

        # central widget
        self.frame = QWidget()
        self.frame.layout = QVBoxLayout()

        self.frame.layout.addWidget(self.top_bar)
        self.frame.layout.addWidget(self.bottom_half, 1)

        self.frame.layout.setContentsMargins(0,0,0,0)
        self.frame.layout.setSpacing(0)
        self.frame.setLayout(self.frame.layout)
        self.setCentralWidget(self.frame)

        self.lip = Lip(lip_queue)
        self.tts=tts


        self.t = QTimer()
        self.t.timeout.connect(self.timer_thread)
        self.t.start(1000)

        self.t2 = QTimer()
        self.t2.timeout.connect(self.repaint)
        self.t2.start(50)

    def timer_thread(self):
        if self.top_bar.timer.min == 0 and self.top_bar.timer.sec == 0:
            self.game_over = "Ran out of time!"
        
        self.top_bar.timer.decrement_time()

    # lip sync stuff


    def paintEvent(self, e):
        painter = QPainter()
        painter.begin(self)

        if self.game_over:

            # painter = QPainter(self)
            painter.setBrush(QColor(0, 0, 0, 80))
            painter.drawRect(400, 300, 300, 100)
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(QPointF(500, 350), self.game_over)

            painter.end()

            self.t.stop()
            self.t2.stop()

            self.side_bar.setVisible(False)

            # painter.setBrush(QColor(0,0,0,50))
            # painter.drawRect(400, 300, 300, 100)
            # painter.drawText(QPointF(500, 400), self.game_over)
            # self.t.stop()
            # self.t2.stop()
            return
        
        painter.setPen(Qt.PenStyle.NoPen)

        image = QImage("robot-3.png").scaled(450,450)
        painter.drawImage(QPointF(375, 200), image)

        lip_img = QImage(self.lip.img_path).scaled(100,100)
        painter.drawImage(QPointF(550, 375), lip_img)

        offset_x = clamp(self.x, 160, 1000) / 50 - 10
        offset_y = clamp(self.y, 140, 600) / 30 - 10

        painter.setBrush(QColor("red"))
        painter.drawEllipse(QPointF(550+offset_x, 310+offset_y), 20, 20)
        painter.drawEllipse(QPointF(650+offset_x, 310+offset_y), 20, 20)

        painter.setBrush(QColor(100,100,200,60))
        if self.shield and time.time() - self.shield < 20:
            painter.drawEllipse(QPointF(620, 300), 300, 300)
        else:
            self.shield = None

        painter.setBrush(QColor(255,100,100))
        targets = []
        for info in self.targets:
            rect = QRect(info[0], info[1], 20,20)
            painter.drawRect(rect)
            targets.append((rect, info))

        painter.setBrush(QColor(100, 200, 100, 80))
        for splat in self.splats:
            rect = QRect(splat[0], splat[1], randint(30,50), randint(30,50))
            painter.drawRect(rect)

            for rect2, info in targets:
                if rect.intersects(rect2):
                    self.targets.remove(info)
                    painter.end()
                    self.repaint()
                    return
            # painter.drawEllipse(QPointF(splat[0], splat[1]), splat[2]*2, splat[2]*2)

        if self.active_path:
            image = QImage(self.active_path).scaled(100,100)
            painter.drawImage(QPointF(900, 50), image)

            pen = QPen(QColor(100, 100, 100, 80), 10)
            painter.setBrush(Qt.GlobalColor.transparent)
            painter.setPen(pen)

            start_point = QPointF(950, 550)
            end_point = QPointF(self.x, self.y)
            control_point = QPointF((start_point.x() + end_point.x()) / 2 + 100, (start_point.y() + end_point.y()) / 2 - 100)
            
            path = QPainterPath()
            path.moveTo(start_point)
            path.cubicTo(start_point, control_point, end_point)

            painter.drawPath(path)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(20,20,20))
        painter.drawRect(0, 0, 160, 600)

        painter.setBrush(QBrush(QPixmap("belt9.png")))
        painter.drawRect(0, 0, 140, 600)

        image = QImage("cannon.png").scaled(200,200)
        painter.drawImage(QPointF(900, 500), image)



            # while True:
            #     time.sleep(1)

        painter.end()

    # @Slot(name="repaint")
    # def update_repaint(self):
    #     self.repaint()

    def voice_thread(self):
        to_send = f"i threw a {' '.join(self.active_path[7:-4].split('_'))} to you, dealing {self.active_damage}%. you now have {self.top_bar.health.value()}/100% health"
        print("sending: "+to_send)
        resp = self.tts.gpt(to_send)
        self.tts.try_speak(resp)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_M and self.active_path and not self.shield:
            # try:
            #     t = threading.Thread(target=self.voice_thread, daemon=True)
            #     t.start()

            # except openai.RateLimitError as e:
            #     self.tts.try_speak("you have to wait")
            #     return
            
            # self.top_bar.health.decrease_health(self.active_damage)

            self.splats.append((self.x, self.y, self.active_damage))
            # for _ in range(20):
            #     self.top_bar.timer.decrement_time()
            # time.sleep(1)
            # resp = "hello thereaksdlfjlkasfjlkdafj ksdfk"

            self.repaint()

            if len(self.targets) == 0:
                if self.top_bar.health.value() - self.active_damage <= 0:
                    self.top_bar.health.setValue(0)
                    self.game_over = "It's dead!"
                    return
                self.top_bar.health.decrease_health(self.active_damage)

                try:
                    t = threading.Thread(target=self.voice_thread, daemon=True)
                    t.start()
                except openai.RateLimitError as e:
                    self.tts.try_speak("you have to wait")
                    return
                    # pass
                
                self.shield = time.time()
                self.active_path = None
                self.active_damage = 0

            self.repaint()
        # if e.key() == Qt.Key.Key_M:
        #     # self.tts.try_speak("testing")
        #         resp = self.tts.gpt("i threw a moldy cheese to you, dealing 20%")
        #         # time.sleep(1)
        #         # resp = "hello thereaksdlfjlkasfjlkdafj ksdfk"
        #         self.tts.try_speak(resp)
        # if e.key() == Qt.Key.Key_Q:
        #     self.lip.img_path = "lip_A.png"
        #     self.repaint()
        # if e.key() == Qt.Key.Key_W:
        #     self.lip.img_path = "lip_B.png"
        #     self.repaint()
            

    # def mouse
    # def mouse(self, e):
    #     print(e.pos())
    #     # pass

    # def eventFilter(self, a, e):
    #     print(a.type())
    #     if e.type() == QEvent.Type.HoverMove:
    #         print("hi")
    #         # print(e.pos())

    #     return QMainWindow.eventFilter(self, a, e)

    def mouseMoveEvent(self, e):
        if self.active_path:
            self.x = e.pos().x()
            self.y = e.pos().y()
            self.repaint()
        # print(e.pos().x(), e.pos().y())

    # def mouseReleaseEvent(self, e):
    #     self.x = 0
    #     self.y = 0

    # def keyPressEvent(self, e):
    #     if e.key() == Qt.Key.Key_W:
    #         self.y -= 5

    #         self.repaint()

    # def hover

    # def event(self, e):
    #     print(e.type())
    #     return super().event(e)


if __name__ == "__main__":
    app = QApplication([])
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    app.exec()

    # while True:
    #     print(window.getm