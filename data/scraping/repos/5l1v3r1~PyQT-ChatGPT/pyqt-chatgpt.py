import sys
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_ChatGPT(object):
    def setupUi(self, ChatGPT):
        ChatGPT.setObjectName("ChatGPT")
        ChatGPT.resize(513, 515)
        ChatGPT.setAutoFillBackground(True)
        p = ChatGPT.palette()
        p.setColor(ChatGPT.backgroundRole(), QtGui.QColor('#12a37f'))
        ChatGPT.setPalette(p)
        self.centralwidget = QtWidgets.QWidget(ChatGPT)
        self.centralwidget.setObjectName("centralwidget")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(10, 130, 491, 31))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(190, 170, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.getInputValue)
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(10, 230, 491, 211))
        self.textEdit.setObjectName("textEdit")
        self.pushButton.clicked.connect(self.getInputValue)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(60, 10, 111, 111))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("../Downloads/chat (1).png"))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(180, 40, 271, 51))
        font = QtGui.QFont()
        font.setPointSize(30)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(390, 460, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(QtCore.QCoreApplication.instance().quit)
        ChatGPT.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(ChatGPT)
        self.statusbar.setObjectName("statusbar")
        ChatGPT.setStatusBar(self.statusbar)

        self.retranslateUi(ChatGPT)
        QtCore.QMetaObject.connectSlotsByName(ChatGPT)
        
    def getInputValue(self):
        query = self.lineEdit.text()
        openai.api_key = "PUT_YOUR_API_KEY_HERE"
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"{query}",
        temperature=1,
        max_tokens=3500,
        top_p=1,
        frequency_penalty=0.2,
        presence_penalty=0
        )
        res = response["choices"][0]["text"]
        self.textEdit.clear()
        self.textEdit.append(f"ChatGPT: {res}")
        
    def retranslateUi(self, ChatGPT):
        _translate = QtCore.QCoreApplication.translate
        ChatGPT.setWindowTitle(_translate("ChatGPT", "ChatGPT"))
        self.lineEdit.setText(_translate("ChatGPT", ""))
        self.pushButton.setText(_translate("ChatGPT", "Send"))
        self.label_2.setText(_translate("ChatGPT", "PyQT ChatGPT"))
        self.pushButton_2.setText(_translate("ChatGPT", "Quit"))


if __name__ == "__main__":
    try:
        import openai
    except ImportError:
        os.system("python3 -m pip install openai")
        import openai

    app = QtWidgets.QApplication(sys.argv)
    ChatGPT = QtWidgets.QMainWindow()
    ui = Ui_ChatGPT()
    ui.setupUi(ChatGPT)
    ChatGPT.show()
    sys.exit(app.exec_())
