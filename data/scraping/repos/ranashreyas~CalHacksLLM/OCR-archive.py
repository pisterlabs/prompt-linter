import os
import subprocess
import pyautogui
import pytesseract
from PIL import ImageGrab
import time

import openai
import json
import requests
from bs4 import BeautifulSoup
import cv2

from hume import HumeBatchClient
from hume.models.config import FaceConfig
from pprint import pprint
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time


# Set up your OpenAI API credentials
openai.api_key = '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'

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
client = HumeBatchClient("bFDAvikKUskwAc1fKS22R7vuoimXeXAIncrBboEq3qp9LhhB")
config = FaceConfig()

# Function to make Google Chrome the active application
def activate_chrome():
    script = """
    tell application "Google Chrome"
        activate
    end tell
    """
    subprocess.call(['osascript', '-e', script])

# Capture screenshot of the active window
def capture_screenshot():
    subprocess.call(['screencapture', '-c', '-x'])

# Convert the clipboard image to text using OCR
def ocr_from_clipboard():
    active_app = subprocess.check_output(['osascript', '-e', 'tell application "System Events" to get name of first process whose frontmost is true'])
    if not active_app.decode('utf-8').startswith('Google Chrome'):
        # print("Please make sure Google Chrome is the active application.")
        return "NOT CHROME"
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