# import constants
# import sys
# import openai
# from PyQt5.QtCore import Qt
# from PyQt5.QtGui import QPixmap
# from PyQt5.QtWidgets import (
#     QApplication,
#     QWidget,
#     QLabel,
#     QLineEdit,
#     QPushButton,
#     QVBoxLayout,
#     QHBoxLayout,
#     QGroupBox, QTextEdit,
# )
#
# openai.api_key = constants.API_KEY
#
#
# class MainWindow(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.init_ui()
#
#     def init_ui(self):
#         # create the widgets
#         self.logo_label = QLabel()
#         self.logo_pixmap = QPixmap('robot.png').scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         self.logo_label.setPixmap(self.logo_pixmap)
#         self.input_label = QLabel('Ask a Question:')
#         self.input_field = QLineEdit
#         self.input_field.setPlaceholderText("Type here ...")
#         self.answer_label = QLabel('Answer :')
#         self.answer_field = QTextEdit()
#         self.answer_field.setReadOnly(True)
#         self.submit_button = QPushButton('submit')
#         self.submit_button.setStyleSheet(
#             """
#             QPushButton{
#                 background-color:#4CAF50;
#                 border:none;
#                 color: white;
#                 padding: 15px 32px ;
#                 font-size: 18px;
#                 font-weight:bold;
#                 border-radius:10 px;
#                 }
#             QPushButton:hover{
#                 background-color: #3e8e41;
#                 }
#                 """
#         )
#
#         self.popular_questions_group = QGroupBox('Popular Questions')
#         self.popular_questions_layout = QVBoxLayout()
#         self.popular_questions = ["What is Machine Learning ?",
#                                   "How do i become Machine learning engineer?",
#                                   "what are the most popular libraries used in machine learning?"]
#         self.question_buttons = []
#
#         # create a layout
#         layout = QVBoxLayout()
#         layout.setContentsMargins(20, 20, 20, 20)
#         layout.setSpacing(20)
#         layout.setAlignment(Qt.AlignCenter)
#
#         # Add Logo
#         layout.addWidget(self.logo_label, alignment=Qt.AlignCenter)
#
#         # Add input field
#         input_layout = QHBoxLayout()
#         input_layout.addWidget(self.input_label)
#         input_layout.addWidget(self.input_field)
#         input_layout.addWidget(self.submit_button)
#         layout.addLayout(input_layout)
#
#         # Add Answer Field
#         layout.addWidget(self.answer_label)
#         layout.addWidget(self.answer_field)
#
#         # Add the popular questions buttons
#
#         for question in self.popular_questions:
#             button = QPushButton(question)
#             button.setStyleSheet(
#                 """
#                 QPushButton{
#                     backgroud-color:#FFFFFF;
#                     border:2px solid #00AEFF;
#                     color:#00AEFF;
#                     padding:10px 20px ;
#                     font-size:18px;
#                     font-weight:bold;
#                     border-radius:5px;
#                     }
#                 QPushButton:hover{
#                     background-color:#00AEFF;
#                     color:#FFFFFF
#                     }"""
#
#             )
#             button.clicked.connect(lambda _, q=question: self.input_field.setText(q))
#             self.popular_questions_layout.addWidget(button)
#             self.question_buttons.append(button)
#         self.popular_questions_group.setLayout(self.popular_questions_layout)
#         layout.addWidget(self.popular_questions_group)
#
#         # set the layout
#         self.setLayout(layout)
#
#         # set the window properties
#         self.setWindowTitle('Real Estate buying Advice')
#         self.setGeometry(200, 200, 600, 600)
#
#         # Connect the sumbint buttonto the function which queries chatgpt
#         self.submit_button.clicked.connect(self.get_answer)
#
#     def get_answer(self):
#         question = self.input_field.text()
#
#         completion = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user",
#                        "contesnt": "you are a Real Estate Agent. Answer the following question in a concise way or with bullet points."},
#                       {"role": "user", "content": f'{question}'}],
#             max_tokens=1024,
#             n=1,
#             stop=None,
#             temperature=0.7,
#
#         )
#
#         answer = completion.choices[0].message.content
#         self.answer_field.setText(answer)
#
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())
