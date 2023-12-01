#be sure to make sh executable

from subprocess import Popen, PIPE
import os
import openai
import pyautogui
import pyperclip
import subprocess
from AnkiConnector import AnkiConnector
import time
from PIL import Image
import glob

def notify(text):
    print('text')    
    msg = "notify-send ' ' '"+text+"'"
    os.system(msg)

def construct_request(prompt, text):    
    item = prompt+text
    return send_request([{"role":"user","content":item}])

def send_request(request_message):
    api_location = '~/projects/grammarpt/apikey.txt'
    api_location = os.path.expanduser(api_location)
    with open(api_location, 'r') as f:
        api_key = f.read().strip()
    openai.api_key = (api_key)
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=request_message
    )
    corrected = response["choices"][0]["message"]["content"].replace("\n","").strip().lstrip()
    notify(corrected)
    pyperclip.copy(corrected)    
    return corrected  

with open('/home/lunkwill/projects/grammarpt/obsidian_dir.txt', 'r') as f:
    obsidian_dir = f.read().strip()

ankti_to_make_location = obsidian_dir+'/ankis_to_make.txt'
ankti_to_make_location = os.path.expanduser(ankti_to_make_location)

card_creation_failed = False
with open(ankti_to_make_location, 'r') as f:
    chunks = f.read().split('\n\n')
    for chunk in chunks:
        if '|' in chunk:
            fact = chunk.split('|')[0]
            if len(fact) < 1000:
                if len(fact) > 10:
                    #notify("Making anki flashcard from the following fact: "+fact)
                    anki_response = construct_request("Make an Anki Flashcard from the following fact. You are free to use your own knowledge to make the card more professional. Label it Front: and Back: .\n\n", fact) 
                    parts = anki_response.split("Front: ")[1].split("Back: ")
                    front, back = [part.strip() for part in parts]
                    url = chunk.split('|')[1]
                    source = url if url else ""
                    #if source doesn't start with http, add it
                    if source and not source.startswith('http'):
                        source = 'http://'+source
                    deck_name = '...MyDiscoveries'
                    note_type = 'Basic'
                    connector = AnkiConnector(deck_name=deck_name, note_type=note_type, allow_duplicate=False)
                    card_creation_failed = not connector.add_card(front, back, source)
                    # if successfully_made:
                    #     with open(ankti_to_make_location, 'r') as mid_make:
                    #         full_file = mid_make.read().split('\n\n')
                    #     full_file.replace(chunk, '')
                    #     with open(ankti_to_make_location, 'w') as replaced:
                    #         replaced.write(full_file)

                else:
                    notify("Too short to be a fact.")
            else:
                notify("Too long for highlight grammar fix. Break it into small parts")

if not card_creation_failed:
#delete all the text in the file
    with open(ankti_to_make_location, 'w') as f:
        f.write('')