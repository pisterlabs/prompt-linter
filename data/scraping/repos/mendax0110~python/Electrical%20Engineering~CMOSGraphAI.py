import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLabel, QLineEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import openai


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set window properties
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle("CMOS Transistor Graphs")

        # Set API key
        openai.api_key = ""

        # Set layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        vbox = QVBoxLayout(central_widget)

        # Set input field
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Enter the length of the channel (in nm): "))
        self.input_field = QLineEdit()
        hbox.addWidget(self.input_field)
        vbox.addLayout(hbox)

        # Set buttons
        self.plot_idvd_button = QPushButton("Plot I_d vs V_d")
        self.plot_idvg_button = QPushButton("Plot I_d vs V_g")
        self.plot_idvdsat_button = QPushButton("Plot I_d vs V_ds_sat")
        vbox.addWidget(self.plot_idvd_button)
        vbox.addWidget(self.plot_idvg_button)
        vbox.addWidget(self.plot_idvdsat_button)

        # Set up Matplotlib figure
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(self.canvas)

        # Set up connections
        self.plot_idvd_button.clicked.connect(lambda: self.plot_data("Id vs Vd", "Vd (V)"))
        self.plot_idvg_button.clicked.connect(lambda: self.plot_data("Id vs Vg", "Vg (V)"))
        self.plot_idvdsat_button.clicked.connect(lambda: self.plot_data("Id vs Vds_sat", "Vds_sat (V)"))

    def plot_data(self, title, xlabel):
        # Get input value
        L = float(self.input_field.text())

        # Set up query
        query = f"plot {title} for a CMOS transistor with channel length {L} nm"

        # Get response from OpenAI API
        response = openai.Completion.create(
            engine="davinci",
            prompt=query,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7,
        )

        # Extract data from response
        data = response.choices[0].text

        # Parse data
        x_data = []
        y_data = []
        for line in data.strip().split("\n"):
            x, y = line.split(",")
            x_data.append(float(x))
            y_data.append(float(y))

        # Plot data
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(x_data, y_data)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Id (A)")
        ax.set_title(title)
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
