import sys
import requests
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout
from PyQt6.QtGui import QGuiApplication
import openai
import json


class CurriculumApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Set window properties
        self.setWindowTitle('Curriculum App')
        self.setGeometry(200, 200, 600, 750)

        screen = QGuiApplication.primaryScreen().geometry()

        # Calculate the center position
        x = int((screen.width() - self.width()) / 2)
        y = int((screen.height() - self.height()) / 2)

        # Set the window position
        self.move(x, y)

        # Create dropdowns
        self.grade_dropdown = QComboBox(self)
        self.grade_dropdown.addItems([str(i) for i in range(1, 8)])
        self.grade_label = QLabel('Grade:', self)

        self.subject_dropdown = QComboBox(self)
        self.subject_dropdown.addItems(['Math', 'Science', 'English', 'Social Studies'])
        self.subject_label = QLabel('Subject:', self)

        self.time_dropdown = QComboBox(self)
        self.time_dropdown.addItems([str(i) for i in [10, 20, 30, 40, 50, 60]])
        self.time_label = QLabel('Time (minutes):', self)

        # Create button and text box
        self.generate_button = QPushButton('Generate Curriculum', self)
        self.generate_button.clicked.connect(self.generate_curriculum)

        self.curriculum_text = QTextEdit(self)

        # Create layouts
        self.vbox = QVBoxLayout()
        self.hbox1 = QHBoxLayout()
        self.hbox2 = QHBoxLayout()

        # Add widgets to layouts
        self.hbox1.addWidget(self.grade_label)
        self.hbox1.addWidget(self.grade_dropdown)
        self.hbox1.addWidget(self.subject_label)
        self.hbox1.addWidget(self.subject_dropdown)
        self.hbox2.addWidget(self.time_label)
        self.hbox2.addWidget(self.time_dropdown)
        self.vbox.addLayout(self.hbox1)
        self.vbox.addLayout(self.hbox2)
        self.vbox.addWidget(self.generate_button)
        self.vbox.addWidget(self.curriculum_text)
        self.setLayout(self.vbox)

    def generate_curriculum(self):
        API_KEY = self.parseApiKey()
        openai.api_key = API_KEY

        # Create query using selected dropdown values
        selected_grade = self.grade_dropdown.currentText()
        selected_subject = self.subject_dropdown.currentText()
        selected_time = self.time_dropdown.currentText()

        query = f"give a curriculum for grade {selected_grade} for subject {selected_subject} that can be conducted for {selected_time} minutes"

        # Send query to OpenAI API and display response
        # For simplicity, here we just display the query itself
        response = self.get_openai_response(query)
        responseStr = response["choices"][0]["text"]
        self.curriculum_text.setText(responseStr)


    def get_openai_response(self, prompt):
        return openai.Completion.create(
                    model="text-davinci-003",
                    prompt= prompt,
                    temperature=0.7,
                    max_tokens=64,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0)

    def parseApiKey(self):
        with open('config.json', 'r') as f:
            config = json.load(f)
            return config['api_key']


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CurriculumApp()
    window.show()
    sys.exit(app.exec())
