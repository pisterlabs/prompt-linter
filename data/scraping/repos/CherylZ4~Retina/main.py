from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from pathlib import Path
import cloudvision
import cohereapi
import photoshoot
import sys

global objCoord

def grabAllDef(initialdict: dict):
    print('Run grabAllDef')
    finaldict = {}
    for keys in initialdict:
        finaldict[keys] = cohereapi.grabDefinition(keys)
        print(keys)
    print(finaldict)
    return finaldict


global current_obj

global file_name 
file_name = 'current.jpg'

stylesheet = ("""
    MainWindow{
        border-image: url(home.jpg) 0 0 0 0 stretch stretch;
        background-repeat: no-repeat; 
        background-position: center;
    }

    #button{
        border-image: url(upload.png) 0 0 0 0 stretch stretch;
        background-repeat: no-repeat; 
        background-position: center;
    }

    #abutton2{
        border-image: url(take_photo.png) 0 0 0 0 stretch stretch;
        background-repeat: no-repeat; 
        background-position: center;
    }

    QLabel{
        border-image: url(define.jpg) 0 0 0 0 stretch stretch;
        background-repeat: no-repeat; 
        background-position: center;
    }

     #info{
        border-image: url(info.png) 0 0 0 0 stretch stretch;
        background-repeat: no-repeat; 
        background-position: center;
    }
""")

global defDict

class ImageWindow(QMainWindow):
    global file_name
    def __init__(self, file_name):
        super().__init__()
        self.setMouseTracking(True)
        self.w = None
        self.size = self.screen().size()
        self.setWindowTitle("Retina")
        # info = QPushButton(self, objectName='info')
        # info.setGeometry(-5, 700, 300, 250)
        # info.setStyleSheet("border-style: outset; border-radius:40px; margin: auto; background-color: #8BC3EB; font-weight:bold; font-size:40px; padding: 15px; width: 75%; font-style:Times New Roman")
        self.showMaximized()
        qss = "ImageWindow{border-image: url(%s) 0 0 0 0 stretch stretch;background-repeat: no-repeat; background-position: center;}" % (
            file_name,
        )
        self.setStyleSheet(qss)
        
    
    def objClicked(self, event):
        global objCoord
        global current_obj
        pos = event.pos()
        for obj in objCoord:
            print(pos.x(), pos.y())
            if objCoord[obj][0][0] * self.size.width() <= pos.x() <= objCoord[obj][2][0] * self.size.width():
                if objCoord[obj][0][1] * self.size.height() <= pos.y() <= objCoord[obj][2][1] * self.size.height():
                    current_obj = obj
                    return True

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Space:
            self.closeEvent()
            # self.close()
            self.w = MainWindow()

        

    def mousePressEvent(self, event):
        pos = event.pos()
        if self.objClicked(event): 
            self.show_new_window(pos.x(), pos.y())

    def show_new_window(self, x, y):
        if self.w is None:
            self.w = Definition(x, y)
            self.w.show()
        else:
            self.w.close()
            self.w = None
    
    def closeEvent(self):
        if self.w:
            self.w.close()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Retina")
        self.setAutoFillBackground(True)
        self.setStyleSheet('background-color: #B9E2FF')
        button = QPushButton(self, objectName='button')
        button.setGeometry(self.height()-5, self.width() - 319, 587, 185)
        button.clicked.connect(self.open_image)
        button.setStyleSheet("border-style: outset; border-radius:40px; margin: auto; background-color: #8BC3EB; font-weight:bold; font-size:40px; padding: 15px; width: 75%; font-style:Times New Roman")
        
        abutton2 = QPushButton(self, objectName='abutton2')
        abutton2.setGeometry(self.height()-5, self.width()-55, 587, 185)
        abutton2.setStyleSheet("border-style: outset; border-radius:40px; margin: auto; background-color: #8BC3EB; font-weight:bold; font-size:40px; padding: 15px; width: 75%; font-style:Times New Roman")
    
        abutton2.clicked.connect(self.take_photo)

        self.showMaximized()
        self.size = self.screen().size()

    def take_photo(self):
        global objCoord
        global defDict
        global file_name
        path = photoshoot.openCam()
        objCoord = cloudvision.grabobjects(path)
        defDict = grabAllDef(objCoord)
        print(objCoord)
        print(self.size.width(), self.size.height())
        if path:
            file_name = path
            self.hide()
            self.image_window = ImageWindow(path)
            self.image_window.show()

    def open_image(self):
        global objCoord
        global defDict
        global file_name
        options = QFileDialog.Option.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Images (*.png *.xpm *.jpg *.bmp);;All Files (*)", options=options)
        # os.rename(old_file_name, new_file_name)
        print(file_name, "!!!!!!!!")
        objCoord = cloudvision.grabobjects(file_name)
        defDict = grabAllDef(objCoord)
        print(objCoord)
        print(self.size.width(), self.size.height())

        if file_name:
            self.hide()
            self.image_window = ImageWindow(file_name)
            self.image_window.show()

class Color(QWidget):

    def __init__(self, color):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)
        self.setStyleSheet('background-color: #DBF0FF')

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor)
        self.setPalette(palette)


class Definition(QWidget):
    
    def __init__(self, x, y):
        super().__init__()
        self.setFixedSize(500,400)
        layout = QVBoxLayout()
        self.setWindowTitle('Definition')

        global current_obj
        global defDict

        self.label = QLabel()
        self.label.setWordWrap(True)

        worddef = defDict[current_obj]
        print(worddef)
        self.label.setText(current_obj + '\n__________\n\n' + worddef)
        self.label.setStyleSheet(
            "font-family: times; "
            "font-size: 20px;"
            "color: #ffffff;"
        )
        
        self.label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.label)

        self.setLayout(layout)



        self.move(x, y)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = MainWindow()
    app.setStyleSheet(stylesheet)
    window.show()
    sys.exit(app.exec())
