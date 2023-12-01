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

def clean_text(text):
    text_file = open("raw.txt", "w")
    n = text_file.write(text)
    text_file.close()

# Main function
def main():
    activate_chrome()

    cap = cv2.VideoCapture(0)
    time.sleep(0.5)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    while True:
        
        # Check if Chrome is the active application
        active_app = subprocess.check_output(['osascript', '-e', 'tell application "System Events" to get name of first process whose frontmost is true'])
        if not active_app.decode('utf-8').startswith('Google Chrome'):
            # print("Please make sure Google Chrome is the active application.")
            continue
        
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


        cv2.imwrite('face.jpg', frame)

        job = client.submit_job(None, [config], files = ['face.jpg'])
        job.await_complete()
        status = job.get_status()
    #     print(f"Job status: {status}")

        details = job.get_details()
        run_time_ms = details.get_run_time_ms()
    #     print(f"Job ran for {run_time_ms} milliseconds")

        predictions = job.get_predictions()

        emotions = []

        try:
            # Extract emotions from the JSON
            for prediction in predictions[0]['results']['predictions']:
                for emotion in prediction['models']['face']['grouped_predictions'][0]['predictions'][0]['emotions']:
                    emotions.append(emotion)

            # Print the extracted emotions
            for emotion in emotions:
                if emotion['name'] == 'Confusion':
                    print(emotion['name'], emotion['score'])
                    if emotion['score'] > 0.4:
                        # Get the active tab's URL and title
                        tab_url = get_active_tab_url()
                        tab_title = get_active_tab_title()
                        print(f"Active Tab: {tab_title} ({tab_url})")
                        
                        # Introduce a 5-second delay
                        # time.sleep(3)
                        
                        # Perform OCR on the active tab's content
                        text = ocr_from_clipboard()
                        
                        if text:
                            # print(f"\nExtracted Text:\n{text}")
                            clean_text(text)
                        else:
                            print("No text was extracted.")
        except: 
            print("no face detected, i think")

if __name__ == '__main__':
    main()
