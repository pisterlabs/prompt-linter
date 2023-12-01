from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextEdit, QLineEdit, QComboBox, QCheckBox, QTabWidget
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QTextCursor
import sys
import sounddevice as sd
from vosk import Model, KaldiRecognizer, SetLogLevel
import json
import openai
import os
from dotenv import load_dotenv
from datetime import datetime

class TranslationThread(QThread):
    signal = pyqtSignal(QTextEdit, str)
    stopped = pyqtSignal(QTextEdit, str)

    def __init__(self, lang_sel, prefer_small_model, enable_logging, text_area, timestamp):
        QThread.__init__(self)
        self.lang_sel = lang_sel
        self.prefer_small_model = prefer_small_model
        self.enable_logging = enable_logging
        self.timestamp = timestamp
        self.stop_flag = False
        self.translation_complete = True
        self.text_area = text_area

    def callback(self, indata, frames, time, status):
        if self.stop_flag:
            raise sd.CallbackAbort
        if status:
            self.signal.emit(str(status))
        else:
            if self.translation_complete:
                self.signal.emit(self.text_area, "\nListening...")
            int_data = (indata[:, 0] * 32767).astype('int16')
            if self.rec.AcceptWaveform(int_data.tobytes()):
                self.translation_complete = False
                self.signal.emit(self.text_area, "\nTranslating...")
                result = json.loads(self.rec.Result())
                if result['text'].strip():
                    self.signal.emit(self.text_area, f"{self.lang}: {result['text']}")
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a professional English translator, adept at translating any language into English while retaining cultural references, puns, and the like. Keep the meaning of your translations in line with the original intent. I want you to only reply with the translated English text: no other information is needed."},
                                {"role": "user", "content": f"Translate the following from {self.lang} to English: {result['text']}"},
                            ],
                        )
                        translated_text = response['choices'][0]['message']['content']
                    except:
                        translated_text = "Translation failed. Apologies for the inconvenience."
                    self.signal.emit(self.text_area, "EN: " + translated_text)
                    if self.enable_logging:
                        with open(f"transcription_log_{self.timestamp}_{self.lang}.txt", "a", encoding='utf-8') as f:  
                            f.write(f"{self.lang}: {result['text']}\n")
                            f.write(f"EN: {translated_text}\n")
                    self.translation_complete = True

    def run(self):
        SetLogLevel(0)
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')
        stream = sd.InputStream(callback=self.callback, channels=1, samplerate=16000, blocksize=65536)
        self.lang = self.lang_sel.upper()
        self.signal.emit(self.text_area, f"Language: {self.lang}")
        if self.prefer_small_model:
            model = Model(f"vosk-model-small-{self.lang}-0.22")
        else:
            model = Model(f"vosk-model-{self.lang}-0.22")
        self.rec = KaldiRecognizer(model, 16000)
        with stream:
            while True:
                pass

    def stop(self):
        self.stop_flag = True
        self.stopped.emit(self.text_area, "Translation stopped")
        if self.enable_logging:
            filename = f"transcription_log_{self.timestamp}_{self.lang}.txt"
            if os.path.isfile(filename) and os.path.getsize(filename) > 0:
                self.stopped.emit(self.text_area, f"Translation saved to log at {filename}")

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Watty's Translations - Alpha - 27 June 2023")

        self.start_button = QPushButton("Start Translation")
        self.start_button.clicked.connect(self.start_translation)

        self.stop_button = QPushButton("Stop Translation")
        self.stop_button.clicked.connect(self.stop_translation)

        self.tab_widget = QTabWidget()

        self.lang_input = QLineEdit()
        self.lang_input.setPlaceholderText("Enter language code")

        self.model_selector = QComboBox()
        self.model_selector.addItem("Prefer Small Model", True)
        self.model_selector.addItem("Prefer Large Model", False)

        self.log_checkbox = QCheckBox("Enable Logging")

        layout = QVBoxLayout()
        layout.addWidget(self.lang_input)
        layout.addWidget(self.model_selector)
        layout.addWidget(self.log_checkbox)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.tab_widget)

        main_widget = QWidget()
        main_widget.setLayout(layout)

        self.setCentralWidget(main_widget)

    def start_translation(self):
        lang_sel = self.lang_input.text()
        prefer_small_model = self.model_selector.currentData()
        enable_logging = self.log_checkbox.isChecked()

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.tab_widget.addTab(self.text_area, lang_sel)

        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.thread = TranslationThread(lang_sel, prefer_small_model, enable_logging, self.text_area, timestamp)
        self.thread.signal.connect(self.update_text)
        self.thread.stopped.connect(self.update_text)
        self.thread.start()

    def stop_translation(self):
        self.thread.stop()

    def update_text(self, text_area, message):
        text_area.moveCursor(QTextCursor.MoveOperation.End)
        text_area.insertPlainText(message + "\n")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec())
