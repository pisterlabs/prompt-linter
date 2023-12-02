#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from Backend.LLM import LLM
import sys
from Backend.chat import getQuestions
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget, QLabel, QFrame
import openai
from dotenv import load_dotenv
import os
class ChatBotApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.prompt = """I want you to act as a Python interpreter. I will type commands and you will reply with what the
        python output should show. I want you to only reply with the terminal output inside one unique
        code block, and nothing else. Do no write explanations, output only what python outputs. Do not type commands unless I
        instruct you to do so. When I need to tell you something in English I will do so by putting
        text inside curly brackets like this: {example text}. My first command is a=1.
        the object is  a {type}, the color of the object is {color}, 
        the material of the object is {material}, and the size of the object is {size}. Output a Python tuple with the object as a String,
        the color as a BGR tuple, the material as a String, and the size of the object in meters as an int in that order. If the user's input
        for any of these parameters is unreasonable, label that slot with the bool False. Do not include any extra words in your answer. 
        If the user's input is not reasonable, write False."""
        self.tuple = []
        self.answer = []
        self.object_q = getQuestions()
        load_dotenv()
        self.questions = self.object_q.get_question()
        self.i = 1
        self.setWindowTitle("Bosser")
        self.setGeometry(100, 100, 400, 400)
        self.setStyleSheet("background-color: #ffffff;")

        self.central_widget = QWidget()


        self.layout = QVBoxLayout()

        self.messages_label = QLabel(f"System: {self.questions[0]}")
        self.messages_label.setStyleSheet("font-size: 18pt")
        self.layout.addWidget(self.messages_label)

        self.user_input = QTextEdit()

        self.layout.addWidget(self.user_input)

        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet("font-size: 18pt;background-color: #588b8b")
        self.send_button.clicked.connect(self.send_message)
        self.layout.addWidget(self.send_button)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.send_message)
        self.next_button.setStyleSheet("background-color: #588b8b")
        self.next_button.setEnabled(False)
        self.layout.addWidget(self.next_button)



        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)
        self.send_button.installEventFilter(self)

    def eventFilter(self, source, event):
        if source is self.user_input and event.type() == Qt.EventType.KeyPress and event.key() == Qt.Key.Enter:
            self.send_message()
            return True
        return super().eventFilter(source, event)


    def send_message(self):
        user_message = self.user_input.toPlainText()
        self.answer = self.ask_question(user_message, self.answer)
        self.messages_label.setText(self.messages_label.text() + f"\nUser: {user_message}")
        self.user_input.clear()
        self.messages_label.setText(self.messages_label.text() + f"\nSystem: {self.questions[self.i]}")
        self.i +=1



    def ask_question(self,input, answer):
        if len(answer) == 3:
            question = self.questions[5] \
                .replace('${type}', answer[0]) \
                .replace('${color}', answer[1]) \
                .replace('${material}', answer[2]) \
                .replace('${size}', input)
            self.questions[5] = question
            answer.append(input)
            return answer
        elif len(answer)== 4:
            set_answer = input
            if set_answer.lower() == 'yes' or set_answer.lower() == 'true':
                self.next_button.setEnabled(True)
                self.next_button.setStyleSheet("font-size: 18pt;background-color: #588b8b")
                self.send_button.setStyleSheet("font-size: 12pt;background-color: #588b8b")
                self.send_button.setEnabled(False)
                self.user_input.setEnabled(False)
                self.user_input.setStyleSheet("background-color: grey")
                return answer
                #object_LLM = LLM(answer)
                #return object_LLM.getanswer()
            else:
                 return []
        else:
                set_answer = input
                if set_answer.lower() != 'yes' and set_answer.lower() != 'true':
                    answer.append(set_answer)
                    return answer
                else:
                    return answer

