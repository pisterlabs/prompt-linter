
import sys

import key # apimiz key.py dosyasında 
import openai # pip install openai
from main import * # uygulamın kaynak kodlarını içeren dosya
from pyperclip import copy as copyy
from PyQt5.QtWidgets import *

app=QApplication(sys.argv)
window=QMainWindow()
ui=Ui_MainWindow()
ui.setupUi(window)
window.show()
window.setWindowTitle("Grammerly")


def translate():
    openai.api_key = key.api_key
    sentences=ui.textEdit.toPlainText()
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Correct this to grammerly English:"+sentences,
        temperature=0,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
        )
    result=response["choices"][0]["text"] # sorguladığımız cümlenin alınması 
    ui.textEdit_2.setText(str([result][0][2:])) # cümlemizin ekrana yazdırılması



def kopyala():
    copyy(ui.textEdit_2.toPlainText())


ui.pushButton.clicked.connect(translate)
ui.pushButton_2.clicked.connect(kopyala)

sys.exit(app.exec_())