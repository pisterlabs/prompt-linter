import sys
import pyttsx3
import speech_recognition as sr
from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit, QShortcut, QMessageBox, QFileDialog
import webbrowser
from dotenv import load_dotenv
import os
import openai
import threading
import re
OPENAI_KEY = None
try:
    load_dotenv()
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    import openai
    openai.api_key = OPENAI_KEY
except:
    print("There is a problem with the OPENAI API key. You cannot use AI mode")

# TO DO: ADD AI mode
class VoiceAssistant(QObject):
    def __init__(self):
        super().__init__()
        self.engine = pyttsx3.init()
        # set a new voice
        #voices = self.engine.getProperty('voices')
        #self.engine.setProperty('voice',voices[10].id)
        self.engine.setProperty('rate', 150)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()


def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        query = recognizer.recognize_google(audio, language='en')
        print(f"User: {query}")
        return query.lower()
    except Exception as e:
        print(f"Sorry, I couldn't understand that: The error is {e}")
        return ""


class VoiceListener(QThread):
    signal_response = pyqtSignal(str)

    def run(self):
        query = listen()
        self.signal_response.emit(query)


class VoiceAssistantApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def closeEvent(self, event):
        self.stop_thread()
        event.accept()

    def init_ui(self):
        self.setWindowTitle('Human-Computer Interaction Assistant')
        self.setGeometry(250, 250, 800, 600)

        self.label = QLabel('Welcome! I am your Human-Computer Interaction Assistant.', self)

        self.text_edit = QTextEdit(self)
        self.text_edit.setPlaceholderText('Ask me something...')
        self.text_edit.setReadOnly(True)
        self.button_listen = QPushButton('Listen', self)
        self.button_listen.clicked.connect(self.on_listen)

        self.button_listen.setStyleSheet(
            "QPushButton {"
            "   background-color: #4CAF50;"
            "   color: white;"
            "   font-size: 16px;"
            "   padding: 8px 16px;"
            "   border: none;"
            "   border-radius: 4px;"
            "   min-width: 100px;"
            "}"
            "QPushButton:hover {"
            "   background-color: lightgreen;"
            "}"
        )
        self.isAI = False
        self.label_ai_mode = QLabel('AI Mode: Closed', self)
        self.button_stop = QPushButton('Stop', self)
        self.button_stop.clicked.connect(self.stop)
        self.button_stop.setShortcut("e")
        self.button_stop.setStyleSheet(
            "QPushButton {"
            "   background-color: red;"
            "   color: white;"
            "   font-size: 16px;"
            "   padding: 8px 16px;"
            "   border: none;"
            "   border-radius: 4px;"
            "   min-width: 100px;"
            "}"
            "QPushButton:hover {"
            "   background-color: #ffcccb;"
            "}"
        )
        self.button_stop.setEnabled(False)

        self.button_save = QPushButton('Save Conversation', self)
        self.button_save.setShortcut("ctrl+s")
        self.button_save.setStyleSheet(
            "QPushButton {"
            "   background-color: #0096FF;"
            "   color: white;"
            "   font-size: 16px;"
            "   padding: 8px 16px;"
            "   border: none;"
            "   border-radius: 4px;"
            "   min-width: 100px;"
            "}"
            "QPushButton:hover {"
            "   background-color: lightblue"
            "}"
        )
        self.button_save.clicked.connect(self.save_conversation)

        self.button_clear = QPushButton('Clear', self)
        self.button_clear.setShortcut("ctrl+c")
        self.button_clear.setStyleSheet(
            "QPushButton {"
            "   background-color: #000000;" 
            "   color: white;"
            "   font-size: 16px;"
            "   padding: 8px 16px;"
            "   border: none;"
            "   border-radius: 4px;"
            "   min-width: 100px;"
            "}"
            "QPushButton:hover {"
            "   background-color: #999999;"
            "}"
        )
        self.button_clear.clicked.connect(self.clear_conversation)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.text_edit)
        self.layout.addWidget(self.button_listen)
        self.layout.addWidget(self.button_stop)
        self.layout.addWidget(self.button_save)
        self.layout.addWidget(self.button_clear)
        self.layout.addWidget(self.label_ai_mode)
        self.setLayout(self.layout)
        

        self.shortcut_space = QShortcut(Qt.Key_Space, self)
        self.shortcut_space.activated.connect(self.on_listen)

        self.listening = False

    def on_listen(self):
        if not self.listening:
            self.listening = True
            self.button_listen.setEnabled(False)
            # change the style sheet of the listen button
            self.button_listen.setStyleSheet(
            "QPushButton {"
            "   background-color: darkgreen;"
            "   color: white;"
            "   font-size: 16px;"
            "   padding: 8px 16px;"
            "   border: none;"
            "   border-radius: 4px;"
            "   min-width: 100px;"
            "}"
            "QPushButton:hover {"
            "   background-color: lightgreen;"
            "}"
        )
            self.button_stop.setEnabled(True)
            self.voice_listener = VoiceListener()
            self.voice_listener.signal_response.connect(self.process_query)
            self.voice_listener.finished.connect(self.on_listen_finished)
            self.voice_listener.start()

    def on_listen_finished(self):
        self.listening = False
        self.button_listen.setEnabled(True)
        self.button_stop.setEnabled(False)
        self.button_listen.setStyleSheet(
            "QPushButton {"
            "   background-color: #4CAF50;"
            "   color: white;"
            "   font-size: 16px;"
            "   padding: 8px 16px;"
            "   border: none;"
            "   border-radius: 4px;"
            "   min-width: 100px;"
            "}"
            "QPushButton:hover {"
            "   background-color: lightgreen;"
            "}"
        )
    
    def stop(self):
        try:
            if self.listening:
                print("Stop Listening...")
                self.voice_listener.terminate()
                self.listening = False
                self.button_listen.setEnabled(True)
                self.button_listen.setStyleSheet(
                "QPushButton {"
                "   background-color: #4CAF50;"
                "   color: white;"
                "   font-size: 16px;"
                "   padding: 8px 16px;"
                "   border: none;"
                "   border-radius: 4px;"
                "   min-width: 100px;"
                "}"
                "QPushButton:hover {"
                "   background-color: lightgreen;"
                "}"
            )
        except Exception as e:
            print(e)
            pass
    def stop_thread(self):
        self.thread.join()

    def process_query(self, query):
        answer = None
        self.text_edit.append(f'User: {query}')
        if "hello" in query:
            answer = "Hello! How can I assist you?"
        elif "goodbye" in query:
            answer = "Goodbye! Have a great day!"
            self.text_edit.append('Assistant: Goodbye! Have a great day!')
            QApplication.quit()
        elif "clear" in query:
            self.clear_conversation()
        elif "save" in query:
            self.save_conversation()
        elif "open chrome" in query or "open google" in query or "new tab" in query:
            webbrowser.open_new_tab("http://www.google.com")
            answer = "I opened Google Chrome."
        elif "open ai" in query or "open artificial intelligence" in query:
            if OPENAI_KEY != None:
                self.isAI = True
                self.label_ai_mode.setText('AI Mode: Open')
                answer = "AI mode is open"
            else:
                answer = "You need to provide a valid API key to use AI mode"
        elif "close ai" in query or "close artificial intelligence" in query:
            self.isAI = False
            self.label_ai_mode.setText("AI Mode: Closed")
            answer = "AI mode is closed"
        else:
            if not self.isAI:
                for question in answers.keys():
                    if question in query.lower():
                        answer = answers[question]
                        break
                    if answer == None:
                        answer = "Sorry, I don't have an answer for that."
            else:
                if query != "" and query is not None:
                    try:
                        answer = self.generate_ai_response(query)
                    except Exception as e:
                        print(e)
                        answer = "AI mode cannot be used right now. Please check your internet connection."
        self.text_edit.append(f'Assistant: {answer}')
        self.thread = threading.Thread(target=voice_assistant.speak, args=(answer,))
        self.thread.start()
    
    def generate_ai_response(self, prompt):
        def eliminate_incomplete_sentences(text):
            # Used regular expression to split the text into sentences. Credit to: https://stackoverflow.com/questions/25735644/python-regex-for-splitting-text-into-sentences-sentence-tokenizing
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

            # Filter out sentences without a period (incomplete sentences)
            complete_sentences = [sentence for sentence in sentences if '.' in sentence]

            # Join the complete sentences back into a string
            result = ' '.join(complete_sentences)
            return result
        response = openai.Completion.create(
            engine = "text-davinci-003",
            prompt = prompt,
            max_tokens = 100,
            n = 1,
            stop = None,
            temperature = 0.5,
            )
        return eliminate_incomplete_sentences(response["choices"][0]["text"])

    def save_conversation(self):
        file_name, _ = QFileDialog.getSaveFileName(self, 'Save Conversation', '', 'Text Files (*.txt)')
        if file_name:
            with open(file_name, 'w') as file:
                file.write(self.text_edit.toPlainText())
            QMessageBox.information(self, 'Save Conversation', 'Conversation saved successfully.')
    def clear_conversation(self):
            self.text_edit.clear()


# Default answers. More answers can be added here.
# credit to https://www.interaction-design.org/literature/topics/human-computer-interaction and https://www.usability.gov/how-to-and-tools/methods/task-analysis.html 
answers = {
    "usability engineering": """
    Usability engineering is a professional discipline that focuses on improving the usability of interactive systems. 
    It draws on theories from computer science and psychology to define problems that occur during the use of such a system. 
    Usability engineering involves the testing of designs at various stages of the development process, with users or with usability experts.
""",
    "human computer interaction": """ Human-computer interaction (HCI) is a multidisciplinary field of study focusing on the design of computer technology and, in particular, the interaction between humans (the users) and computers. While initially concerned with computers, HCI has since expanded to cover almost all forms of information technology design. """,

    "task analysis": """Task analysis is the process of learning about ordinary users by observing them in action to understand in detail how they perform their tasks and achieve their intended goals. 
    Tasks analysis helps identify the tasks that your website and applications must support and can also help you refine or re-define your site’s navigation or search by determining the appropriate content scope."""
}


if __name__ == '__main__':
    app = QApplication(sys.argv)
    voice_assistant = VoiceAssistant()
    window = VoiceAssistantApp()
    # Show the manual at the beginning of the program
    manual = (
        "Welcome to the Human-Computer Interaction Assistant!\n"
        "Ask me questions about usability engineering, human-computer interaction, "
        "task analysis in HCI, or simply say 'hello' or 'goodbye'.\n"
        "I can also open Google Chrome.\n"
        "You can open AI mode to get answers powered by artificial intelligence.\n"
        "To say something, click the 'Listen' button and speak.\n"
        "I will answer your questions, and you can also save the conversation to a text file."
    )

    window.text_edit.append(manual)
    window.show()
    sys.exit(app.exec_())