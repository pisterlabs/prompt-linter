import sys
import logging
import traceback

from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtCore import QSize, QThreadPool, qFatal
from api import OpenAIChat
from custom_widgets import *
import random
import os
import uuid
import datetime
import pickle
import csv
import json
import pathlib


logging.basicConfig(level=logging.ERROR, filename="log.txt", filemode="a")
app = QApplication([])

if "key.json" not in os.listdir():
    key = {"api_key": "None", "model": "gpt-4"}
    with open("key.json", 'w') as key_file:
        json.dump(key, key_file)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('QDR Chat')
        size = app.primaryScreen().size()
        self.resize(QSize(2*size.width()//3, 2*size.height()//3))
        self.setWindowIcon(QIcon('icons/ai.ico'))

        self.main_container = QHBoxLayout()
        self.main_container.setSpacing(0)
        self.main_container.setContentsMargins(0, 0, 0, 0)
        self.main_widget = QWidget()
        self.main_widget.setObjectName("main_widget")
        self.main_widget.setLayout(self.main_container)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        with open("style/style.qss") as f:
            self.setStyleSheet(f.read())

        with open("style/config.json") as f:
            config = json.load(f)
        message_size = config["message_font_size"]
        code_size = config["code_font_size"]
        input_size = config["input_font_size"]

        self.history_widget = LeftWidget(message_size, code_size, input_size)
        self.chat_widget = ChatWidget(message_size, code_size, input_size)
        self.chat_widget.about(True)

        self.get_history()
        self.history_widget.new_chat_btn.clicked.connect(self.new_session)
        self.history_widget.history_list.itemPressed.connect(self.load_session)
        self.history_widget.size_dialog.signal.FontSizesChanged.connect(self.change_text_size)

        self.main_container.addWidget(self.history_widget, 1)
        self.main_container.addWidget(self.chat_widget, 5)

        self.chat_widget.input_widget.send_button.clicked.connect(self.send)
        self.chat_widget.system_message_widget.signal.SystemMessageChanged.connect(self.edit_system_message)
        self.history_widget.history_list.clicked.connect(self.show_about)

        self.current_session = None

        self.pool = QThreadPool()
        self.oldPos = self.pos()
        pywinstyles.change_header_color(self, color="#202123")
        pywinstyles.change_border_color(self, color="#515473")

        self.setCentralWidget(self.main_widget)

    def send(self):
        if self.chat_widget.input_widget.text_edit.toPlainText() == '':
            return
        message = self.chat_widget.input_widget.text_edit.toPlainText()
        if self.current_session is None:
            title = "New Chat"
            self.current_session = Session(title)
            self.current_session.system_message = self.chat_widget.system_message_widget.system_message.text()
            self.history_widget.add(title)
            self.show_about(False)
            self.get_title(message, self.current_session, self.history_widget.history_list.currentItem())

        self.current_session.append_message(
            message, 'user'
        )
        self.chat_widget.message_display_widget.add_message(
            message, 'user'
        )
        self.run_model(self.current_session)
        QTimer.singleShot(5, self.chat_widget.scroll_to_bottom)
        self.chat_widget.input_widget.text_edit.setText('')
        self.current_session.save()

    def get_title(self, query, session, item):
        with open("key.json", 'r') as api_key:
            file = json.load(api_key)
            api_key, model = file['api_key'], file['model']
            query_title_thread = OpenAIChat(api_key, model, session, query)
            query_title_thread.signals.result.connect(lambda x: self.change_title(x, session, item))
            self.pool.start(query_title_thread)

    @staticmethod
    def change_title(title, session, item):
        if isinstance(title, str):
            new_title = "API Error"
        else:
            new_title = title.content.strip('"')
        session.title = new_title
        item.setText(new_title)
        item.setToolTip(new_title)
        with open('history/history.csv', 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        for n, (id_, _, old_title) in enumerate(rows):
            if id_ == session.id_:
                rows[n] = [id_, _, new_title]
                break
        with open('history/history.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def edit_system_message(self, text):
        if self.current_session is not None:
            self.current_session.system_message = text
            self.current_session.save()

    def run_model(self, session):
        with open("key.json", 'r') as api_key:
            file = json.load(api_key)
            api_key, model = file['api_key'], file['model']
        llm = OpenAIChat(api_key, model, session)
        llm.signals.started.connect(self.set_disable)
        llm.signals.result.connect(lambda x: self.receive(x, session))
        self.pool.start(llm)

    def set_disable(self):
        self.chat_widget.toggle_loading_indicator()

    def receive(self, result, session):
        if isinstance(result, str):
            ErrorDialog().message(result)
        else:
            session.append_message(result.content, 'assistant')
            if session.id_ == self.current_session.id_:
                self.chat_widget.message_display_widget.add_message(result.content, 'assistant')
        QTimer.singleShot(5, self.chat_widget.scroll_to_bottom)
        self.chat_widget.input_widget.text_edit.setText('')
        self.set_disable()
        session.save()

    def load_session(self, _):
        self.chat_widget.about(False)
        self.chat_widget.clear()
        with open(f'history/history.csv', 'r') as f:
            reader = csv.reader(f)
            index = self.history_widget.history_list.currentIndex()
            id_ = list(reader)[index.row()][0]
        with open(f'history/{id_}.chs', 'rb') as f:
            self.current_session = pickle.load(f)
        self.chat_widget.set_system_message(self.current_session.system_message)
        self.chat_widget.display_messages(self.current_session.messages)
        self.chat_widget.scroll_to_bottom()

    def get_history(self):
        if 'history.csv' not in os.listdir('history'):
            pathlib.Path("history/history.csv").touch()
        with open('history/history.csv', 'r') as csvfile:
            csv_reader = list(csv.reader(csvfile))
            if csv_reader is not None:
                for row in reversed(csv_reader):
                    title = row[2]
                    self.history_widget.add(title)
        self.history_widget.history_list.clearSelection()

    def new_session(self):
        self.chat_widget.about(True)
        self.chat_widget.clear()
        self.chat_widget.set_system_message("You are a helpful assistant")
        self.history_widget.history_list.setCurrentItem(None)
        self.current_session = None

    def show_about(self, item):
        if item is None:
            self.chat_widget.about(True)
        else:
            self.chat_widget.about(False)

    def change_text_size(self, message_size, code_size, input_size):
        self.chat_widget.set_font_size(message_size, code_size, input_size)
        self.history_widget.set_current_font_size(message_size, code_size, input_size)
        with open("style/config.json", "w") as f:
            sizes = {"message_font_size": message_size,
                     "code_font_size": code_size,
                     "input_font_size": input_size}
            json.dump(sizes, f)
        if self.history_widget.history_list.selectedItems():
            self.load_session(None)

    def closeEvent(self, event):
        sys.exit()


class Session:
    def __init__(self, title):
        self.title = title
        self.start = datetime.datetime.now()
        self.system_message = "You are a helpful assistant"
        self.messages = []
        self.id_ = str(uuid.uuid4())

    def append_message(self, message, role):
        self.messages.append({"role": role, "message": message})

    def save(self):
        if self.id_ + '.chs' not in os.listdir('history'):
            with open('history/history.csv', 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
            rows.insert(0, [self.id_, self.start, self.title])
            with open('history/history.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
        with open(f'history/{self.id_}.chs', 'wb') as cls:
            pickle.dump(self, cls)


def except_hook(type_, value, traceback_):
    exception = traceback.TracebackException(type_, value, traceback_)
    for frame_summary in exception.stack:
        frame_summary.filename = os.path.relpath(frame_summary.filename)
    logging.exception("".join(exception.format()))
    qFatal('')


sys.excepthook = except_hook
window = MainWindow()
window.show()
app.exec()
