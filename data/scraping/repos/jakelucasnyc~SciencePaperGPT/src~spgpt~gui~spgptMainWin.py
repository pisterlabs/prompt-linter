import sys
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QListWidget, QListView, QPushButton, QFileDialog, QListWidgetItem, QTextEdit
from spgpt.pdf import retrieve_pdf_data
from spgpt.query import get_response_from_query
from spgpt.gui.importPdf import ImportPDF
from spgpt.gui.query import Query
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import logging

class SPGPTMainWin(QMainWindow):

    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)

    def __init__(self, embeddings:OpenAIEmbeddings, cache_dir:str):
        super().__init__()
        self._threadpool = QThreadPool()
        self._embeddings = embeddings
        self._cache_dir = cache_dir
        self._faiss_dict = {}
        

        # Initialize widgets and layouts
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('PDF Chat with GPT')

        main_layout = QHBoxLayout()

        # Left Column
        left_column = QVBoxLayout()

        import_button = QPushButton('Import PDF')
        import_button.clicked.connect(self.import_pdf)
        # import_button.setFixedWidth(40)
        import_button.setStyleSheet("QPushButton { background-color: #42b1f5; color: white; font: bold; }")
        left_column.addWidget(import_button, alignment=Qt.AlignTop)

        self.pdf_list = QListWidget()
        self.pdf_list.setAcceptDrops(True)
        self.pdf_list.setDragEnabled(True)
        self.pdf_list.setDropIndicatorShown(True)
        self.pdf_list.setSelectionMode(QListView.ExtendedSelection)
        self.pdf_list.setDragDropMode(QListView.InternalMove)
        left_column.addWidget(self.pdf_list)

        # Middle Column
        middle_column = QVBoxLayout()
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        middle_column.addWidget(self.chat_display)

        input_layout = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText('Type your question here...')
        input_layout.addWidget(self.prompt_input)

        self.send_button = QPushButton('Submit')
        self.send_button.setStyleSheet("QPushButton { background-color: #42b1f5; color: white; font: bold; }")
        input_layout.addWidget(self.send_button)
        middle_column.addLayout(input_layout)
        self.send_button.clicked.connect(lambda: self.on_user_input())

        self.creativity_line_edit = QLineEdit()
        self.creativity_line_edit.setText("0.3")
        self.creativity_line_edit.setFixedWidth(50)
        input_layout.addWidget(self.creativity_line_edit)

        # Right Column
        right_column = QVBoxLayout()

        new_conversation_button = QPushButton('New Conversation')
        new_conversation_button.setStyleSheet("QPushButton { background-color: #42b1f5; color: white; font: bold; }")
        right_column.addWidget(new_conversation_button, alignment=Qt.AlignTop)

        self.conversations_list = QListWidget()
        right_column.addWidget(self.conversations_list)
        new_conversation_button.clicked.connect(self.add_new_conversation)


        # Set main layout
        main_widget = QWidget()
        main_layout.addLayout(left_column, 1)
        main_layout.addLayout(middle_column, 2)
        main_layout.addLayout(right_column, 1)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Set dark theme
        self.set_dark_theme()

        # Start in full-screen mode
        self.showFullScreen()

    def add_new_conversation(self):
        new_conversation = QListWidgetItem("New Conversation")
        self.conversations_list.addItem(new_conversation)
        self.chat_display.clear()

    def import_pdf(self):
        file_dialog = QFileDialog(self, "Import PDF", "", "PDF Files (*.pdf)")
        file_dialog.setAcceptMode(QFileDialog.AcceptOpen)
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        if file_dialog.exec() == QFileDialog.Accepted:
            file_path = file_dialog.selectedFiles()[0]
            self.pdf_list.insertItem(0, f'Processing: {os.path.basename(file_path)}')
            pdf_item = self.pdf_list.item(0)
            worker = ImportPDF(file_path, self._embeddings, self._cache_dir)
            # worker.finished.connect()
            worker.data_acquired.connect(lambda faiss_db: self._faiss_dict.update({file_path: faiss_db}))
            worker.data_acquired.connect(lambda: pdf_item.setText(file_path))
            worker.error.connect(lambda: pdf_item.setText(f'Error: {file_path}'))
            self._threadpool.start(worker)

    def _get_temperature(self) -> float:
        temp_input = self.creativity_line_edit.text()
        try:
            temp = float(temp_input)
        except Exception as e:
            self._logger.warning(f'Invalid temperature input: {temp_input}. Must be a float between 0 and 1')
            return 0.3

        if not (0 <= temp <= 1):
            self._logger.warning(f'Invalid temperature input: {temp}. Must be a float between (or equal to) 0 and 1')
            return 0.3
        return temp
        
        

    def on_user_input(self):
        user_input = self.prompt_input.text()
        if not user_input:
            return
        # print input to textedit
        self.chat_display.append(f'User: {user_input}\n')
        # clear user input
        self.prompt_input.clear()
        # get selected pdf documents
        selectedDocs = self.pdf_list.selectedItems()
        if not selectedDocs:
            self.chat_display.append(f'System: Please highlight a document before making a prompt.\n')
            return
        doc = selectedDocs[0]
        faiss_db = self._faiss_dict[doc.text()]
        temp = self._get_temperature()
        print(f'Temperature: {temp}')

        #disabling input widgets for when AI is coming up with response
        self.prompt_input.setEnabled(False)
        self.send_button.setEnabled(False)
        # run query function with relevant faiss_db(s)
        worker = Query(faiss_db, user_input, temperature=temp)
        worker.signals.error.connect(lambda e: self.chat_display.append(f'AI: {e}\n'))
        worker.signals.response_acquired.connect(lambda response: self.chat_display.append(f'AI: {response}\n'))
        worker.signals.finished.connect(lambda: self.send_button.setEnabled(True))
        worker.signals.finished.connect(lambda: self.prompt_input.setEnabled(True))
        self._threadpool.start(worker)


    def set_dark_theme(self):
        dark_palette = QPalette()

        # Base colors
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Base, QColor(42, 42, 42))
        dark_palette.setColor(QPalette.AlternateBase, QColor(66, 66, 66))
        dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))

        # Selection colors
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

        # Set the application palette
        QApplication.setPalette(dark_palette)