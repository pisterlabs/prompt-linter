import sys
import subprocess
import openai
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QLabel,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QComboBox,
    QFrame,
    QScroller,
    QScrollerProperties,
)
from PyQt6.QtCore import Qt

# Set up OpenAI GPT API credentials
openai.api_key = 'YOUR OPENAI API KEY GOES HERE'

class NmapGPTGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nmap GPT GUI")

        # Set the title color to green
        self.setStyleSheet('*[dialog=true] QLabel { color: green; font-size: 14pt; font-weight: bold; }')

        # Top Widget
        self.top_frame = QFrame()
        self.top_frame.setFrameShape(QFrame.Shape.NoFrame)
        self.top_frame.setStyleSheet('background-color: black;')
        self.setCentralWidget(self.top_frame)

        self.layout = QVBoxLayout(self.top_frame)

        self.user_label_gpt = QLabel("You:")
        self.user_label_gpt.setStyleSheet('color: green; font: bold;')
        self.layout.addWidget(self.user_label_gpt)

        self.entry_field_gpt = QLineEdit()
        self.entry_field_gpt.setStyleSheet('color: green; background-color: black;')
        self.entry_field_gpt.returnPressed.connect(self.generate_instructions)
        self.layout.addWidget(self.entry_field_gpt)

        self.text_area_gpt = QTextEdit()
        self.text_area_gpt.setReadOnly(True)
        self.text_area_gpt.setStyleSheet('color: green; background-color: black;')
        self.layout.addWidget(self.text_area_gpt)

        self.reply_button_gpt = QPushButton("Reply to Morpheus")
        self.reply_button_gpt.setStyleSheet('color: green; background-color: black;')
        self.reply_button_gpt.clicked.connect(self.generate_instructions)
        self.layout.addWidget(self.reply_button_gpt)

        self.clear_button_gpt = QPushButton("Clear GPT Text")
        self.clear_button_gpt.setStyleSheet('color: green; background-color: black;')
        self.clear_button_gpt.clicked.connect(self.clear_gpt_text)
        self.layout.addWidget(self.clear_button_gpt)

        # Separator
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.Shape.HLine)
        self.layout.addWidget(self.separator)

        # Bottom Widget
        self.bottom_frame = QFrame()
        self.bottom_frame.setFrameShape(QFrame.Shape.NoFrame)
        self.bottom_frame.setStyleSheet('background-color: black;')
        self.layout.addWidget(self.bottom_frame)

        self.layout_bottom = QVBoxLayout(self.bottom_frame)

        self.nmap_label = QLabel("Nmap:")
        self.nmap_label.setStyleSheet('color: green; font: bold;')
        self.layout_bottom.addWidget(self.nmap_label)

        # Define a list of pre-populated nmap commands
        self.nmap_commands = [
            'nmap -sS',  # SYN scan
            'nmap -sV',  # Version scan
            'nmap -sP',  # Ping scan
            'nmap -sT',  # TCP connect scan
            'nmap -A',  # All-ports scan
            'nmap -O',  # OS detection scan
            'nmap -v',  # Verbose scan
            'nmap -p',  # Port scan
            'nmap -iL', # Read hosts/IPs
             # from a file
        ]

        self.nmap_combo_box = QComboBox()
        self.nmap_combo_box.setStyleSheet('color: green; background-color: black;')
        self.nmap_combo_box.addItems(self.nmap_commands)
        self.layout_bottom.addWidget(self.nmap_combo_box)

        self.entry_field_nmap = QLineEdit()
        self.entry_field_nmap.setStyleSheet('color: green; background-color: black;')
        self.entry_field_nmap.returnPressed.connect(self.run_nmap_command)
        self.layout_bottom.addWidget(self.entry_field_nmap)

        self.text_area_nmap = QTextEdit()
        self.text_area_nmap.setReadOnly(True)
        self.text_area_nmap.setStyleSheet('color: green; background-color: black;')
        self.layout_bottom.addWidget(self.text_area_nmap)
    
    def configure_scrolling_behavior(self):
        QScroller.grabGesture(self.text_area_gpt, QScroller.LeftMouseButtonGesture)
        QScroller.grabGesture(self.text_area_nmap, QScroller.LeftMouseButtonGesture)

        scroller_properties = {
        QScrollerProperties.DragVelocitySmoothingFactor: 0.6,
        QScrollerProperties.MinimumVelocity: 0.0,
        QScrollerProperties.MaximumVelocity: 0.5,
        QScrollerProperties.AcceleratingFlickMaximumTime: 0.4,
        QScrollerProperties.AcceleratingFlickSpeedupFactor: 1.2,
        QScrollerProperties.SnapPositionRatio: 0.2,
        QScrollerProperties.MaximumClickThroughVelocity: 0,
        QScrollerProperties.DragStartDistance: 0.001,
        QScrollerProperties.MousePressEventDelay: 0.5
    }

        QScrollerProperties().setScrollMetric(QScrollerProperties.ScrollMetric.OvershootDragResistanceFactor, 0.5)
        QScrollerProperties().setScrollMetric(QScrollerProperties.ScrollMetric.OvershootDragDistanceFactor, 0.1)
        QScrollerProperties().setScrollMetric(QScrollerProperties.ScrollMetric.OvershootScrollDistanceFactor, 0.1)
        QScrollerProperties().setScrollMetric(QScrollerProperties.ScrollMetric.OvershootScrollTime, 0.2)
        QScrollerProperties().setScrollMetric(QScrollerProperties.ScrollMetric.SnapPositionRatio, 0.2)
        QScrollerProperties().setScrollMetric(QScrollerProperties.ScrollMetric.MaximumSnapTime, 0.3)

        QScroller.scroller(self.text_area_gpt).setScrollerProperties(scroller_properties)
        QScroller.scroller(self.text_area_nmap).setScrollerProperties(scroller_properties)

        self.run_button_nmap = QPushButton("Run Nmap Command")
        self.run_button_nmap.setStyleSheet('color: green; background-color: black;')
        self.run_button_nmap.clicked.connect(self.run_nmap_command)
        self.layout_bottom.addWidget(self.run_button_nmap)

        self.clear_button_nmap = QPushButton("Clear Nmap Text")
        self.clear_button_nmap.setStyleSheet('color: green; background-color: black;')
        self.clear_button_nmap.clicked.connect(self.clear_nmap_text)
        self.layout_bottom.addWidget(self.clear_button_nmap)

        self.configure_scrolling_behavior()

    def generate_instructions(self):
        input_text = self.entry_field_gpt.text()
        self.text_area_gpt.append(f"You: {input_text}")

        try:
            messages = [
                {"role": "system", "content": "You are Morpheus, a helpful assistant."},
                {"role": "user", "content": input_text},
        ]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=200
        )

            generated_text = response['choices'][0]['message']['content']
            self.text_area_gpt.append(f"Morpheus: {generated_text}\n")
        except Exception as e:
            self.text_area_gpt.append(f"Morpheus: Error: {str(e)}\n")

        self.entry_field_gpt.clear()

    def run_nmap_command(self):
        nmap_command = self.nmap_combo_box.currentText()
        user_input = self.entry_field_nmap.text()
        command = f"{nmap_command} {user_input}"
        self.entry_field_nmap.clear()

        try:
            output = subprocess.check_output(command, shell=True).decode()
            self.text_area_nmap.append(f"Morpheus: {output}\n")
        except subprocess.CalledProcessError as e:
            self.text_area_nmap.append(f"Morpheus: {e}\n")

    def clear_gpt_text(self):
        self.text_area_gpt.clear()

    def clear_nmap_text(self):
        self.text_area_nmap.clear()

    def configure_scrolling_behavior(self):
        text_area_gpt_scroller = QScroller.scroller(self.text_area_gpt.viewport())
        text_area_gpt_scroller.grabGesture(self.text_area_gpt.viewport(), QScroller.GestureType.TouchGesture)

        text_area_nmap_scroller = QScroller.scroller(self.text_area_nmap.viewport())
        text_area_nmap_scroller.grabGesture(self.text_area_nmap.viewport(), QScroller.GestureType.TouchGesture)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    nmap_gpt_gui = NmapGPTGUI()
    nmap_gpt_gui.show()

    sys.exit(app.exec())