
import openai
import os
import PyPDF2
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QPushButton, QTextEdit, QLabel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up OpenAI API credentials
        openai.api_key = 'TYPE_YOUR_API_KEY'

        # Create UI elements
        self.select_file_button = QPushButton('Select PDF File', self)
        self.select_file_button.move(20, 20)
        self.select_file_button.clicked.connect(self.select_file)

        self.summary_label = QLabel('Summary:', self)
        self.summary_label.move(20, 60)

        self.summary_text_edit = QTextEdit(self)
        self.summary_text_edit.setReadOnly(True)
        self.summary_text_edit.setGeometry(20, 80, 600, 400)

        self.setWindowTitle('PDF Summarizer')
        self.setGeometry(100, 100, 640, 480)

    def select_file(self):
        # Open file dialog to select PDF file
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter('PDF Files (*.pdf)')
        file_dialog.setWindowTitle('Select PDF File')
        if file_dialog.exec_() == QFileDialog.Accepted:
            # Read text from selected PDF file
            file_path = file_dialog.selectedFiles()[0]
            try:
                with open(file_path, mode='rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text()
            except FileNotFoundError:
                self.summary_text_edit.setPlainText('The specified file could not be found.')
            except PyPDF2.PdfStreamError:
                self.summary_text_edit.setPlainText('The specified file could not be read.')

            # Generate summary using OpenAI API
            if text:
                model = 'text-davinci-002'
                prompt = f'Summarize the contents of this PDF file: "{text}"'
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    max_tokens=150,
                    n=1,
                    stop=None,
                    temperature=0.5,
                )

                # Display summary in text edit widget
                summary = response.choices[0].text
                self.summary_text_edit.setPlainText(summary)
            else:
                self.summary_text_edit.setPlainText('There was an error processing the file.')


if __name__ == '__main__':
    # Create and show main window
    app = QApplication([])
    main_window = MainWindow()
    main_window = MainWindow()
    app.exec_()
   
    
    
    


