import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLineEdit, QWidget, QLabel, QTextEdit, QFormLayout, QHBoxLayout, QComboBox, QGridLayout
from PyQt5.QtGui import QPixmap
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
import openai
import allocation
import analysis
import financials
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter
import copy

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setGeometry(400, 400 ,400, 600)

        # Create a horizontal layout for the logo and other widgets
        self.top_layout = QHBoxLayout()

        # Inserting change logo at top
        self.logo_label = QLabel(self)
        pixmap = QPixmap('logo.png')
        self.logo_label.setPixmap(pixmap)
        self.logo_label.setScaledContents(True)
        self.logo_label.setFixedSize(450, 200)
        self.top_layout.addWidget(self.logo_label)

        # Create a vertical layout for the input fields and buttons
        self.layout = QVBoxLayout()
        self.layout.setSpacing(5)


        # Age input field
        self.age_layout = QFormLayout()
        self.age_label = QLabel("Age:")
        self.age_field = QLineEdit()
        self.age_field.editingFinished.connect(self.update_age)
        self.age_layout.addRow(self.age_label, self.age_field)
        self.layout.addLayout(self.age_layout)

        # Deposit Balance input field
        self.deposit_layout = QFormLayout()
        self.deposit_label = QLabel("Balance Deposit:")
        self.deposit_field = QLineEdit()
        self.deposit_field.editingFinished.connect(self.update_deposit)
        self.deposit_layout.addRow(self.deposit_label, self.deposit_field)
        self.layout.addLayout(self.deposit_layout)

        # Salary input field
        self.salary_layout = QFormLayout()
        self.salary_label = QLabel("Salary:")
        self.salary_field = QLineEdit()
        self.salary_field.editingFinished.connect(self.update_salary)
        self.salary_layout.addRow(self.salary_label, self.salary_field)
        self.layout.addLayout(self.salary_layout)

        #Plan option drop down
        # Plan option drop down
        self.plan_layout = QFormLayout()
        self.plan_label = QLabel("Plan:")
        self.risk_level_combobox = QComboBox()
        self.risk_level_combobox.addItem("Default")
        self.risk_level_combobox.addItem("High Risk Long Term")
        self.risk_level_combobox.addItem("Low Risk Long Term")
        self.risk_level_combobox.addItem("High Risk Short Term")
        self.risk_level_combobox.addItem("Low Risk Short Term")
        self.risk_level_combobox.currentIndexChanged.connect(self.update_risk)
        self.plan_layout.addRow(self.plan_label, self.risk_level_combobox)
        self.layout.addLayout(self.plan_layout)

        

        # Chat bot button
        self.chat_button = QPushButton("Go to Chat", self)
        self.chat_button.setStyleSheet('QPushButton {border: 1px solid black;}')
        self.chat_button.clicked.connect(self.open_chat_window)



        # Budget screen button
        self.button = QPushButton("Go to Budget", self)
        self.button.setStyleSheet('QPushButton {border: 1px solid black;}')
        self.button.clicked.connect(self.open_budget_window)

        # Add buttons to the layout
        self.layout.addWidget(self.chat_button)
        self.layout.addWidget(self.button)

        # Add the top_layout and layout to a QVBoxLayout
        self.main_layout = QVBoxLayout()
        self.main_layout.addLayout(self.top_layout)
        self.main_layout.addLayout(self.layout)

        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

        app.setStyle('Oxygen')


    def update_deposit(self):
        deposit = int(self.deposit_field.text())
        allocation.deposit = deposit

    def update_risk(self, index):
        selected_option = self.risk_level_combobox.itemText(index)
        allocation.user['Plan'] = selected_option

    def update_age(self):
        age = int(self.age_field.text())
        allocation.user['Age'] = age

    def update_salary(self):
        salary = int(self.salary_field.text())
        allocation.user['Salary'] = salary
        #allocation.balance['Liquid'] = salary * 0.4
        


    def open_budget_window(self):
        self.budget_window = BudgetWindow()
        self.budget_window.show()
    

    def open_chat_window(self):
        self.chat_window = ChatWindow()
        self.chat_window.show()


class BudgetWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

        self.update_button = QPushButton("Update Chart")
        self.layout.addWidget(self.update_button)

        self.currentBalanceFigure = Figure(figsize=(10, 16))
        self.canvas = FigureCanvasQTAgg(self.currentBalanceFigure)
        self.layout.addWidget(self.canvas)

        self.next_month_button = QPushButton("Next Month")
        self.layout.addWidget(self.next_month_button)

        self.balance_allocation_figure = Figure(figsize=(10, 16))
        self.balance_allocation_canvas = FigureCanvasQTAgg(self.balance_allocation_figure)
        self.layout.addWidget(self.balance_allocation_canvas)

        self.forecasted_return_button = QPushButton("Forecasted Return")
        self.layout.addWidget(self.forecasted_return_button)

        self.forecast_figure = Figure(figsize=(10, 16))
        self.forcast_canvas = FigureCanvasQTAgg(self.forecast_figure)
        self.layout.addWidget(self.forcast_canvas)

        self.update_button.clicked.connect(self.update_chart)
        self.next_month_button.clicked.connect(self.next_month_chart)
        self.forecasted_return_button.clicked.connect(self.update_forecast)

        self.setLayout(self.layout)
        self.setMinimumHeight(900)

    month_counter = 1 


    def update_chart(self):
        allocation.allocate(allocation.balance, allocation.deposit, allocation.user)

        balance_data = list(allocation.balance.values())
        balance_labels = list(allocation.balance.keys())

        self.currentBalanceFigure.clear()
        ax = self.currentBalanceFigure.add_subplot(111)
        bars_current = ax.bar(balance_labels, balance_data)
        ax.set_xlabel('Balance Types')
        ax.set_ylabel('Amount')
        ax.set_title('Current Balance Distribution')
        ax.set_xticklabels(balance_labels, rotation=0)
        self.currentBalanceFigure.tight_layout()
        
        for bar in bars_current:
            height = bar.get_height()
            rounded_height = round(height, 2)
            formatted_height = f'${rounded_height:,.2f}'
            ax.text(bar.get_x() + bar.get_width() / 2, height / 2, formatted_height, ha='center', va='bottom')

        self.balance_allocation_figure.clear()
        ax = self.balance_allocation_figure.add_subplot(111)
        bars_allocation = ax.bar(balance_labels, balance_data)
        ax.set_xlabel('Balance Types')
        ax.set_ylabel('Amount')
        ax.set_title('Balance Allocation After '+ str(self.month_counter) + ' Month(s)')
        ax.set_xticklabels(balance_labels, rotation=0)
        self.balance_allocation_figure.tight_layout()  # Add tight_layout here

        for bar in bars_allocation:
            height = bar.get_height()
            rounded_height = round(height, 2)
            formatted_height = f'${rounded_height:,.2f}'
            ax.text(bar.get_x() + bar.get_width() / 2, height / 2, formatted_height, ha='center', va='bottom')

        self.canvas.draw()
        self.balance_allocation_canvas.draw()

    def next_month_chart(self):
        self.month_counter+=1
        balance_dict = analysis.compound_allocate(allocation.balance, allocation.user)
        balance_data = list(balance_dict.values())
        balance_labels = list(balance_dict.keys())

        self.balance_allocation_figure.clear()
        ax = self.balance_allocation_figure.add_subplot(111)
        bars = ax.bar(balance_labels, balance_data)
        ax.set_xlabel('Balance Types')
        ax.set_ylabel('Amount')
        ax.set_title('Balance Allocation After '+ str(self.month_counter) + ' Month(s)')
        ax.set_xticklabels(balance_labels, rotation=0)
        self.balance_allocation_figure.tight_layout()  # Add tight_layout here

        for bar in bars:
            height = bar.get_height()
            rounded_height = round(height, 2)
            formatted_height = f'${rounded_height:,.2f}'
            ax.text(bar.get_x() + bar.get_width() / 2, height / 2, formatted_height, ha='center', va='bottom')

        self.canvas.draw()
        self.balance_allocation_canvas.draw()
    
    def update_forecast(self):
        self.tempdict = {"Liquid": 1, "ST Fixed Income": 1.05, "LT Fixed Income": 1.042, "ETF": 1.09, "Tech": financials.TOP25_ROI + 1, "CurrRetirement": 1.06}
        year_interest = {}

        for key in allocation.balance.keys():
            allocation.balance[key] = allocation.balance[key] * self.tempdict[key]
        
        allocation.user['Age']+=1
        allocation.allocate(allocation.balance, allocation.user['Salary'], allocation.user)

        balance_data = list(allocation.balance.values())
        balance_labels = list(allocation.balance.keys())

        self.forecast_figure.clear()
        ax = self.forecast_figure.add_subplot(111)
        bars_current = ax.bar(balance_labels, balance_data)
        ax.set_xlabel('Balance Types')
        ax.set_ylabel('Amount')
        ax.set_title('Balance Distribution After Year of Interest + A Year of Salary')
        ax.set_xticklabels(balance_labels, rotation=0)
        self.forecast_figure.tight_layout()
        
        for bar in bars_current:
            height = bar.get_height()
            rounded_height = round(height, 2)
            formatted_height = f'${rounded_height:,.2f}'
            ax.text(bar.get_x() + bar.get_width() / 2, height / 2, formatted_height, ha='center', va='bottom')

        self.forcast_canvas.draw()
        

class ChatWindow(QWidget):
    current_messages = []
    with open("allocation.py", "r") as file:
        code_contents = file.read()
    prompt  = (f"You are a talented financial assistant, answering financial literacy questions with expert ability. You can provide information on how to make smart financial decision, and information on financial institutions in general. Limit all your responses to a few paragraphs AT MOST, you want to express the information in a clear and concise way, while limiting the amount of characters used. Here are the contents of my algorithm that distributes inputted money:\n\n{code_contents}\n\n. The current balance distribution is: " + str(allocation.balance) +". If users ask about the logistics behind the allocation of funds, do not explain the algorithmic details, but explain the theory behind it. For example, its better to invest more riskly for long-term when your young, and viceversa when your old, explaining how our algorithm is somewhat of a gradient. Remeber, you are a financial expert and know why the algorithm makes the financial decisions it does, defend these decisions. Do not try to explicitly describe how the algorithm works, rather the financial reasoning why. Remember to keep your answers brief but clear and effedtive, trying to save")
    current_messages.append({"role":"system", "content": prompt})
    
    
    def __init__(self):
        super().__init__()
        
        self.setGeometry(100, 100, 400, 400)

        self.layout = QVBoxLayout()

        self.chat_log = QTextEdit()
        self.chat_log.setReadOnly(True)

        self.user_input = QLineEdit()
        self.user_input.returnPressed.connect(self.update_chat)

        self.layout.addWidget(self.chat_log)
        self.layout.addWidget(self.user_input)

        self.setLayout(self.layout)

        openai.api_key = 'insert api key here'

        self.chat_log.append("Meet our AI-powered financial advisor, willing to answer any and all financial questions!")
        self.chat_log.append(' ')

    def update_chat(self):
       
        self.current_messages.append({'role': 'user', 'content': "The current balance distribution is " + str(allocation.balance) +"."})

        user_message = self.user_input.text()
        self.chat_log.append(f'<strong>User:</strong> {user_message}\n\n')
        self.chat_log.append(' ')


        # creating message dict
        message_as_dict = {"role": "user", "content": user_message}

        # adding new message as dict to list of messages
        self.current_messages.append(message_as_dict)

        # creating chat model with updated list of messages
        completion = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = self.current_messages
        )

        # getting model reply
        reply = completion.choices[0].message

        # updating list of messages with model's reply
        self.current_messages.append(reply)

        # returning model's reply
    
        self.chat_log.append(f'<strong>&cent;hange Advisor:</strong> {reply["content"]}\n\n')
        self.chat_log.append(' ')
        self.user_input.clear()



app = QApplication(sys.argv)
main_window = MainWindow()
main_window.show()
sys.exit(app.exec_())

