
import os
import sys
import json
import random
import shutil
import openai
from PyQt5.QtWidgets import (
        QApplication, QWidget, QLineEdit, QPushButton, QTextEdit, QLabel, QShortcut, QFileDialog, QMessageBox
    )
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import QSize, QRect
from PyQt5.QtGui import * 
from PIL import Image

API = ["cen", 
       "cen", 
       "cen", 
       "cen"
       ]

openai.api_key = random.choice(API)

question = ''
json_file = open('.history.json', 'a')

class GPT(QWidget):
    
    
    
    def __init__(self, title = " "):
        super().__init__() 
        self.title = title
        self.width = 700
        self.height = 700
        self.widget()

        self.setWindowIcon(QIcon('logo.ico'))

        #if (os.path.isfile('.background.jpg')):

        if(os.path.isfile('.background.jpg')):
            self.setStyleSheet("background-color : #1e1e1e; color: #d4d4d4;")
            self.main_place.setStyleSheet(f"""

                                color: #d4d4d4;  
                                background: url('.background.jpg');
                                background-repeat: repeat-y; 
                                width: {self.size().width()}px;
                                height: {self.size().height()}px;

                                """)
        else:
            self.setStyleSheet("""
                                color: #d4d4d4; 
                                background-color: #1e1e1e;
                                font-family: 'Montserrat', sans-serif;
                               
                                """)


    def widget(self):
        
        self.i = 0
        self.setWindowTitle(self.title)

        self.main_place = QTextEdit(self)
        self.main_place.resize(self.size().width()+65, self.size().height()+20)
        self.main_place.move(10, 10)
        self.main_place.setFont(QFont('Arial', 15))

#################################################
        

        self.question = QLineEdit(self)
        self.question.move(10, self.size().height()+50)
        self.question.resize(self.size().width(), 55)
        self.question.setFont(QFont('Arial', 15))

        self.button = QPushButton(self, text = "Submit")
        self.button.move(self.size().width()+20, self.size().height()+50)
        self.button.resize(55, 55)
        self.button.clicked.connect(self.question_changed)
        self.question.returnPressed.connect(self.question_changed)

        try:
            self.history = open('.history.json', 'r')
            self.main_place.append(self.history.read() + '\n')
        except Exception:
            pass  

        self.theme = QShortcut(QKeySequence("Ctrl+G"), self)
        self.theme.activated.connect(self.set_theme)

        self.background = QShortcut(QKeySequence("Ctrl+O"), self)
        self.background.activated.connect(self.set_background)

        self.del_history = QShortcut(QKeySequence("Ctrl+Shift+D"), self)
        self.del_history.activated.connect(self.history_del)

        self.show()
 
    def history_del(self):
        json_file.close()
        self.history.close()
        self.main_place.clear()
        try:
            os.remove('.history.json')
        except Exception:
            pass
    

    def set_theme(self):
        
        if(self.i % 2 == 0):
            self.setStyleSheet("""color: black; 
                                background-color: white;
                                """)
        else:
            self.setStyleSheet("""color: #d4d4d4; 
                            background-color: #1e1e1e;
                                      
                            font-family: 'Montserrat', sans-serif;
                            """)
        self.i += 1
        

    def set_background(self):

        self.back = open('.background.json', 'w')

        self.wb_patch = QFileDialog.getOpenFileName()[0]
        self.back.writelines(self.wb_patch)
        img = Image.open(self.wb_patch)
        self.new_image = img.resize((self.size().width(), self.size().height()))

        self.new_image.save('.background.jpg')

        self.setStyleSheet
        ("""
            color: black;  
            background: url('.background.jpg'); 
            background-repeat: repeat-y; 
            font-family: 'Montserrat', sans-serif;
        """)

        
      
    # start the app

    @pyqtSlot()
    def question_changed(self):

        self.main_place.resize(self.size().width()-15, self.size().height()-75)
        self.question.resize(self.size().width()-75, 55)
        self.question.move(10, self.size().height()-60)
        self.button.move(self.size().width()-60, self.size().height()-60)     

        
        
        if (os.path.isfile('.background.jpg') and os.path.isfile('.background.json')):

            self.back = open('.background.json', 'r').readline()
            print(self.back)
            print(self.size().width(), self.size().height())
            img = Image.open(f"{self.back}")

            self.new_image = img.resize((self.size().width(), self.size().height()))
            self.new_image.save('.background.jpg')

        if(os.path.isfile('.background.jpg')):

            self.setStyleSheet("background-color : #1e1e1e; color: #d4d4d4;")
            self.main_place.setStyleSheet(f"""

                                color: #d4d4d4;  
                                background: url('.background.jpg');
                                background-repeat: repeat-y; 
                                width: {self.size().width()}px;
                                height: {self.size().height()}px;

                                """)
        else:
            self.setStyleSheet("""
                                color: #d4d4d4; 
                                background-color: #1e1e1e;
                                font-family: 'Montserrat', sans-serif;
                               
                                """)


        user = self.question.text().lower()

        self.main_place.append("You >>> " + user)
        self.question.clear()
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": user}
                ]
            )
            answer = str(completion.choices[0].message)

            translation = json.loads(answer)
            self.main_place.append("\n" + "ChatGPT-3.5 >>> " + translation['content'] + '\n')
        
        
            json_file.writelines("\n" + "You >>> " + user)
            json_file.writelines("\n" + "ChatGPT-3.5 >>> " + translation['content'] + '\n')
            
        except Exception: 
            openai.api_key = random.choice(API) 

def main():
    app = QApplication(sys.argv)
    w = GPT(title = "GPT-3.5")
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
