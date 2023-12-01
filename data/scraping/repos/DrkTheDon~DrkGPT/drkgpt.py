#!/usr/bin/python3

#####################################################
## DrK ChatGPT Writer (DrkGPT)                     ##
## We do a little bit of trolling.                 ##
## https://drkbro.ml/                              ##
## Coded by: drk, DragonSlayer64                   ##
#####################################################

import openai
import os
import time
import requests
import json
from pynput.keyboard import Key, Controller
import random

# Main Defines
def clearcmd():
    os.system('cls' if os.name == 'nt' else 'clear')
keyboard = Controller()

# Main Variable
openai.api_key = "YOUR_API_KEY"
url = "https://api.openai.com/v1/completions"
clearcmd()
prompt = input("DrkGPT\n>")
clearcmd()
print("Generating Answer, this might take some time.")
# API REQUEST
headers = {"Authorization": f"Bearer {openai.api_key}"}
data = {'model': 'text-davinci-003', 'prompt': f'{prompt}', "temperature": 0, "max_tokens": 2048}
raw_json = requests.post(url, headers=headers, json=data).json()

# Getting generated text from json and stripping it out of the \n lines.
plain_text = raw_json['choices'][0]['text']
plain_text_stripped = plain_text.strip() # Finished Completion

# UI 
clearcmd()
print("Successfully generated text.")
time.sleep(2)
clearcmd()



# Code for typing out the text
fast = input("How fast do you want to type the text in?\nFAST = 1\nMEDIUM = 2\nSLOW = 3\n\nOption\n>")
clearcmd()
input("Get ready to switch to the documents tab!\nPress Enter to continue")
print("You have 5 seconds to get ready to switch to Document. STAY ON TAB UNTIL ITS DONE TYPING!")
time.sleep(5)

if fast == "1":
    def write_text():
        for i in plain_text_stripped:
            pause = random.randint(1,200)
            if pause == 21:
                print("Waiting for 15 seconds, this is a function that will make ur writing more trustworthy")
                time.sleep(15)
                pass
            else:
                pass
            timesleep = random.uniform(0.01,0.15)
            keyboard.press(i)
            keyboard.release(i)
            time.sleep(timesleep)


elif fast == "2":
    def write_text():
        for i in plain_text_stripped:
            pause = random.randint(1,200)
            if pause == 21:
                print("Waiting for 15 seconds, this is a function that will make ur writing more trustworthy")
                time.sleep(15)
                pass
            else:
                pass
            timesleep = random.uniform(0.01,0.25)
            keyboard.press(i)
            keyboard.release(i)
            time.sleep(timesleep)

elif fast == "3":
    def write_text():
        for i in plain_text_stripped:
            pause = random.randint(1,200)
            if pause == 21:
                print("Waiting for 15 seconds, this is a function that will make ur writing more trustworthy")
                time.sleep(15)
                pass
            else:
                pass
            timesleep = random.uniform(0.01,0.50)
            keyboard.press(i)
            keyboard.release(i)
            time.sleep(timesleep)

else:
    print("Wrong option, Either choose 1 or 2. Quitting.")
    quit()


write_text()


