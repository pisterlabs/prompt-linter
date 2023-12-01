import sys
import requests
import json
import openai
from PyQt5.QtWidgets import QApplication, QMainWindow, QLineEdit, QTextEdit, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QScrollArea
from PyQt5.QtGui import QIcon, QPixmap, QTextCursor, QTextOption


class SearchAnswer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Create the search bar
        self.search_bar = QLineEdit()
        search_label = QLabel("Ask:")

        # Create the answer box
        self.answer_box = QTextEdit()
        self.answer_box.setReadOnly(True)
        self.answer_box.setWordWrapMode(QTextOption.NoWrap)
        answer_label = QLabel("Answer:")

        # Connect the search bar to the function that makes the API request
        self.search_bar.returnPressed.connect(self.make_request)

        # Create the vertical layout for the search bar and answer box
        v_layout = QVBoxLayout()

        # Create the horizontal layout for the search bar
        h_layout_search = QHBoxLayout()
        h_layout_search.addWidget(search_label)
        h_layout_search.addWidget(self.search_bar)

        # Create the horizontal layout for the answer box
        h_layout_answer = QHBoxLayout()
        h_layout_answer.addWidget(answer_label)

        # Create the scroll area for the answer box
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.answer_box)
        scroll_area.setWidgetResizable(True)

        h_layout_answer.addWidget(scroll_area)

        # Add the search bar and answer box to the vertical layout
        v_layout.addLayout(h_layout_search)
        v_layout.addLayout(h_layout_answer)

        # Create the central widget and set its layout
        central_widget = QWidget()
        central_widget.setLayout(v_layout)
        self.setCentralWidget(central_widget)

        # Set the window title and size
        self.setWindowTitle("ChatGPT PyQt5 v0.0.1")
        self.resize(600, 400)

        # Create the ChatGPT icon
        icon = QIcon(QPixmap("chatgpt-icon.png"))
        self.setWindowIcon(icon)

    def make_request(self):
        # Get the text from the search bar
        search_text = self.search_bar.text()

        # Make the API request
        openai.api_key = "your api key here"
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt="User: " + search_text,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        ).choices[0].text

        # Check if the request was successful
        if "```" in response:

            self.answer_box.setHtml("<pre>" + response + "</pre>")
        else:
            # Clear the search bar
            self.search_bar.clear()

            # Set the response text in the answer box
            cursor = self.answer_box.textCursor()
            self.answer_box.setText(response)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SearchAnswer()
    window.show()
    sys.exit(app.exec_())
