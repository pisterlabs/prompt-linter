import os
import sys
import subprocess
import openai
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QPushButton, QFileDialog, QComboBox, QWidget, QMessageBox, QInputDialog, QProgressDialog)
from PyQt5.QtCore import QSettings, Qt
from PyQt5 import sip

from lxml import etree as ET  # Use lxml's ElementTree API
import html

def install_module(module_name):
    """Install the given module using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", module_name])

# List of modules to check
required_modules = ["PyQt5", "openai", "lxml"]

for module in required_modules:
    try:
        __import__(module)
    except ImportError:
        print(f"{module} not found. Installing...")
        install_module(module)

openai.api_key = "Your OPENAI_API_KEY"

class CustomTableWidgetItem(QTableWidgetItem):
    def __init__(self, text, file_path, label):
        super().__init__(text)
        self.file_path = file_path
        self.label = label

    def __hash__(self):
        return hash((self.file_path, self.label))

    def __eq__(self, other):
        if isinstance(other, CustomTableWidgetItem):
            return self.file_path == other.file_path and self.label == other.label
        return False


class TranslationApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Translation Tool")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.language_box = QComboBox(self)
        self.languages = ["English", "French", "German", "Russian", "Spanish", "Italian", "Portuguese", "Polish", "Korean", "Japanese", "Chinese"]
        self.language_box.addItems(self.languages)
        ##self.language_box.currentIndexChanged.connect(self.switch_language)

        self.table = QTableWidget(self)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['File Name', 'Label', 'String'])

        
        self.table.itemChanged.connect(self.on_item_changed)

        self.load_button = QPushButton("Load English XML File", self)
        self.load_button.clicked.connect(self.load_single_file)

        self.save_button = QPushButton("Save Translations", self)
        self.save_button.clicked.connect(self.save_translations)

        self.translate_button = QPushButton("Translate", self)
        self.translate_button.clicked.connect(self.perform_translation)
        layout.addWidget(self.load_button)
        layout.addWidget(self.translate_button)
        layout.addWidget(self.language_box)
        layout.addWidget(self.table)
        layout.addWidget(self.save_button)
        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        self.english_strings = {}
        self.changed_items = set()
        self.parent_directory = ""
        self.settings = QSettings("Stardock", "GC4Translator")
        last_directory = self.settings.value("last_directory", "H:\\Projects\\GC4\\GalCiv4\\Game\\Data\\English")
        if os.path.exists(last_directory):
            self.parent_directory = last_directory

        # Add the OpenAI Key button
        self.openai_key_button = QPushButton("Enter OpenAI Key", self)
        self.openai_key_button.clicked.connect(self.set_openai_key)
        layout.addWidget(self.openai_key_button)

        # Retrieve and set the openai key if it exists
        openai_key = self.settings.value("openai_key", None)
        if openai_key:
            openai.api_key = openai_key


    def load_single_file(self):
        # Clear the table and the english_strings dictionary
        self.table.setRowCount(0)
        self.english_strings.clear()
        
        starting_directory = self.parent_directory if self.parent_directory else "H:\\Projects\\GC4\\GalCiv4\\Game\\Data\\English"
        file_path, _ = QFileDialog.getOpenFileName(self, "Select XML File", starting_directory, "XML Files (*.xml)")
        
        if file_path:
            self.parent_directory = os.path.dirname(file_path)
            self.settings.setValue("last_directory", file_path)

            file_name = os.path.basename(file_path)
            tree = ET.parse(file_path)
            for elem in tree.findall('StringTable'):
                label = elem.find('Label').text
                string = elem.find('String').text
                self.english_strings[(file_name, label)] = (string, file_path)  # Store with filename and label

            self.populate_table()


    def set_openai_key(self):
        # Open an input dialog to get the OpenAI key
        key, ok = QInputDialog.getText(self, 'OpenAI Key', 'Enter your OpenAI key:')
        if ok:
            # Set the key in the openai library
            openai.api_key = key
            # Store the key using QSettings
            self.settings.setValue("openai_key", key)
            # Optional: You can display a message saying the key has been set
            QMessageBox.information(self, "Success", "OpenAI Key has been set successfully!")

    
    def on_item_changed(self, item):
        if item.column() == 2:  # Check if the translation column was changed
            self.changed_items.add(item)

    def save_translations(self):
        import html

        if not self.changed_items:
            QMessageBox.information(self, "Info", "No changes detected to save.")
            return

        # Extract file_path from the first item (since all items should have the same file path)
        file_path = next(iter(self.changed_items)).file_path

        parser = ET.XMLParser(remove_blank_text=True)
        tree = ET.parse(file_path, parser)
        
        for item in self.changed_items:
            # Check if the item is still valid
            if sip.isdeleted(item):
                continue

            for elem in tree.findall('StringTable'):
                if elem.find('Label').text == item.label:
                    fixed_text = html.unescape(item.text())
                    elem.find('String').text = fixed_text

        try:
            with open(file_path, "wb") as f:
                f.write(ET.tostring(tree, pretty_print=True, encoding='utf-8', xml_declaration=True))
            # Inform the user that save was successful
            QMessageBox.information(self, "Success", f"Translations saved successfully to {file_path}")
        except PermissionError:
            QMessageBox.warning(self, "Error", f"Permission denied when trying to write to {file_path}")
            return
        except Exception as e:  # Catch any other exceptions
            QMessageBox.warning(self, "Error", f"Error when trying to write to {file_path}: {str(e)}")
            return

        self.changed_items.clear()



    def populate_table(self):
        self.table.setRowCount(len(self.english_strings))
        self.table.itemChanged.disconnect(self.on_item_changed)

        for idx, ((file_name, label), (string, file_path)) in enumerate(self.english_strings.items()):
            self.table.setItem(idx, 0, CustomTableWidgetItem(file_name, file_path, label))
            self.table.setItem(idx, 1, CustomTableWidgetItem(label, file_path, label))
            self.table.setItem(idx, 2, CustomTableWidgetItem(string, file_path, label))
        
        self.table.itemChanged.connect(self.on_item_changed)



    def translate_to_language(self, text, target_language):
        prompt = f"Translate this English string without using more words into {target_language}: {text}"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                n=1,
                stop=None,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in getting translation feedback: {e}")
            return None

    def perform_translation(self):
        selected_rows = list(set(item.row() for item in self.table.selectedItems()))
        total_rows = len(selected_rows)

        # Create a QProgressDialog
        if total_rows > 4:
            progress = QProgressDialog("Please Wait...", None, 0, total_rows, self)
            progress.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
            progress.setWindowModality(Qt.WindowModal)  # This will block the main window
            progress.show()

        for idx, row in enumerate(selected_rows):
            english_text_item = self.table.item(row, 2)  # Corrected column index for English
            english_text = english_text_item.text()

            target_language = self.language_box.currentText()
            translated_text = self.translate_to_language(english_text, target_language)

            # Check if translated_text is None (error occurred during translation)
            if translated_text is None:
                QMessageBox.warning(self, "Translation Error", f"Failed to translate the string: {english_text}")
                continue  # Skip the current loop iteration

            translation_item = CustomTableWidgetItem(translated_text, english_text_item.file_path, english_text_item.label)
            self.table.setItem(row, 2, translation_item)  # Overwrite the existing English text with the translation

            # Update the button text to show progress
            self.translate_button.setText(f"Translating {idx + 1} of {total_rows} entries")
            # Update the progress dialog
            if total_rows > 4:
                progress.setValue(idx + 1)
            QApplication.processEvents()  # To update the UI immediately

        # Reset the button text after translation
        self.translate_button.setText("Translate")
        # Close the progress dialog after the operation
        if total_rows > 4:
            progress.close()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = TranslationApp()
    mainWin.show()
    sys.exit(app.exec_())
