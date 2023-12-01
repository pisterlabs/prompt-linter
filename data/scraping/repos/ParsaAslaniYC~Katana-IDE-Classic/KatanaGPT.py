import sys
from PyQt6 import QtWidgets
import openai

# قرار دادن API key در این متغیر
openai.api_key = "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"

# تابعی برای ارسال پیام به ChatGPT و دریافت پاسخ
def send_message(message):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=message,
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text.strip()

class ChatWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.input_label = QtWidgets.QLabel("Enter message:")
        self.input_textbox = QtWidgets.QLineEdit()
        self.send_button = QtWidgets.QPushButton("Send")
        self.output_label = QtWidgets.QLabel("AI:")
        self.output_textbox = QtWidgets.QTextEdit()
        self.output_textbox.setReadOnly(True)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.input_label)
        layout.addWidget(self.input_textbox)
        layout.addWidget(self.send_button)
        layout.addWidget(self.output_label)
        layout.addWidget(self.output_textbox)

        self.setLayout(layout)
        self.send_button.clicked.connect(self.send_message)

    def send_message(self):
        message = self.input_textbox.text()
        response = send_message(message)
        self.output_textbox.setText(response)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())
