from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QLineEdit, QTextEdit
import sys
import os
import subprocess
import pyautogui
import pytesseract
from PIL import ImageGrab
import time
import openai

# Set up your OpenAI API credentials
openai.api_key = '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'

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

# Define the global variable to store the additional prompt text
stuff = ""

# Function to make Google Chrome the active application
def activate_chrome():
    script = """
    tell application "Google Chrome"
        activate
    end tell
    """
    subprocess.call(['osascript', '-e', script])

def activate_gui():
    script = """
    tell application "python"
        activate
    end tell
    """
    subprocess.call(['osascript', '-e', script])

# Capture screenshot of the active window
def capture_screenshot():
    subprocess.call(['screencapture', '-c', '-x'])

# Convert the clipboard image to text using OCR
def ocr_from_clipboard():
    capture_screenshot()
    clipboard_image = ImageGrab.grabclipboard()
    if clipboard_image is not None:
        clipboard_image.save('screenshot.png', 'PNG')
        text = pytesseract.image_to_string('screenshot.png')
        os.remove('screenshot.png')
        return text.strip()
    return None

# Get the active Chrome tab's URL
def get_active_tab_url():
    script = """
    tell application "Google Chrome"
        get URL of active tab of front window
    end tell
    """
    url = subprocess.check_output(['osascript', '-e', script])
    return url.decode('utf-8').strip()

# Get the active Chrome tab's title
def get_active_tab_title():
    script = """
    tell application "Google Chrome"
        get title of active tab of front window
    end tell
    """
    title = subprocess.check_output(['osascript', '-e', script])
    return title.decode('utf-8').strip()

def generate_response(prompt):
    # Define the parameters for the completion
    model = 'text-davinci-003'  # Choose the model you want to use
    max_tokens = 500  # Adjust the desired length of the generated response

    # Generate the completion
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
    )

    # Extract and return the generated text
    completion_text = response.choices[0].text.strip()
    return completion_text

# Main function
def main():
    activate_chrome()
    # Check if Chrome is the active application
    active_app = subprocess.check_output(['osascript', '-e', 'tell application "System Events" to get name of first process whose frontmost is true'])
    if not active_app.decode('utf-8').startswith('Google Chrome'):
        print("Please make sure Google Chrome is the active application.")
        return

    # Get the active tab's URL and title
    tab_url = get_active_tab_url()
    tab_title = get_active_tab_title()
    # print(f"Active Tab: {tab_title} ({tab_url})")
    
    # Introduce a 5-second delay
    # time.sleep(5)
    
    # Perform OCR on the active tab's content
    text = ocr_from_clipboard()
    
    if text:
        text_input = text
        # Provide your prompt here
        prompt = "analyze the page and perform the following action: " + stuff + 'here is the context:' + text

        # Generate a response based on the prompt
        response = generate_response(prompt)

        # Print the generated response
        print(response)
        return response
    else:
        print("No text was extracted.")
        return None


# GUI code
app = QApplication(sys.argv)
window = QWidget()
layout = QVBoxLayout()

label_input = QLabel("Enter additional text for prompt:")
textbox_input = QLineEdit()
button = QPushButton("Run Script")

label_output = QLabel("Output:")
textbox_output = QTextEdit()
textbox_output.setReadOnly(True)  # Make the output text box read-only

layout.addWidget(label_input)
layout.addWidget(textbox_input)
layout.addWidget(button)
layout.addWidget(label_output)
layout.addWidget(textbox_output)
window.setLayout(layout)


def run_script():
    global stuff  # Access the global variable
    stuff = textbox_input.text()  # Update the additional text

    result = main()  # Call the main function
    if result is not None:
        textbox_output.setText(result)  # Update the output text box with the result
    else:
        textbox_output.setText("No result.")


button.clicked.connect(run_script)

window.show()
sys.exit(app.exec_())
