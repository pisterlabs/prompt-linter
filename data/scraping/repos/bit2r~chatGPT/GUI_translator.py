import openai
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit

openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the translation function
def translate_text(text, source_language, target_language):
    prompt = f"Translate the following '{source_language}' text to '{target_language}': {text}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )

    translation = response.choices[0].message.content.strip()
    return translation

class MultilingualTranslationTool(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Multilingual Translation Tool")

        layout = QVBoxLayout()

        # Text input
        text_label = QLabel("Text:")
        self.text_input = QLineEdit()

        # Source language input
        source_lang_label = QLabel("Source Language:")
        self.source_lang_input = QLineEdit()

        # Target language input
        target_lang_label = QLabel("Target Language:")
        self.target_lang_input = QLineEdit()

        # Translate button
        translate_button = QPushButton("Translate")
        translate_button.clicked.connect(self.on_translate_click)

        # Result label
        self.result_label = QLabel()

        layout.addWidget(text_label)
        layout.addWidget(self.text_input)
        layout.addWidget(source_lang_label)
        layout.addWidget(self.source_lang_input)
        layout.addWidget(target_lang_label)
        layout.addWidget(self.target_lang_input)
        layout.addWidget(translate_button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def on_translate_click(self):
        text = self.text_input.text()
        source_language = self.source_lang_input.text()
        target_language = self.target_lang_input.text()

        translated_text = translate_text(text, source_language, target_language)
        self.result_label.setText(translated_text)

if __name__ == "__main__":
    app = QApplication([])
    window = MultilingualTranslationTool()
    window.show()
    app.exec_()
