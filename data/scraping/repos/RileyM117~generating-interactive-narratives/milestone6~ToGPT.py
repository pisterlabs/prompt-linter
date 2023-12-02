import tkinter as tk
from tkinter import simpledialog
import openai
import os
import time

print("start ShowMenu()",flush=True)

# Send a command to Camelot.
def action(command):
    print('start ' + command)
    while(True):
        i = input()
        if(i == 'succeeded ' + command):
            return True
        elif(i == 'failed ' + command):
            return False
        elif(i.startswith('error')):
            return False
# Set up a small house with a door.
action('CreatePlace(Camp, Camp)')
action('CreateCharacter(Bob, B)')
action('SetClothing(Bob, Peasant)')
action('SetPosition(Bob, Camp)')
action('CreateCharacter(Jim, B)')
action('SetClothing(Jim, Peasant)')
action('SetPosition(Jim, Camp.LeftLog)')
action('EnableIcon("Talk to", Talk, Jim, "Talk to")')
action('ShowMenu()')

# Respond to input.
while(True):
  conversations= []
  i = input()
  if(i == 'input Selected Start'):
   action('SetCameraFocus(Bob)')
   action('HideMenu()')
   action('EnableInput()')
   action('EnableInput()')
  if(i == 'input Talk to Jim'):
    ROOT = tk.Tk()
    ROOT.withdraw()
    ask = ("What's your name?")
    action("SetDialog("+ask+")") 
    action('ShowDialog()')
    USER_INP = simpledialog.askstring(title="Test", prompt="What's your Name?:")
    answer = str("Give me a story about " + USER_INP)
    conversations.append(answer)
    action('SetDialog('+answer+')') 
    action('ShowDialog()')
    os.environ["OPENAI_API_KEY"]="input key here!"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    story = ""
    while story == "":
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "user", "content":str(conversations) }
            ]
        )
        story = completion.choices[0].message.content
    action("SetDialog(\""+story+"\")")
    action("ShowDialog()")
    time.sleep(10)
    action("HideDialog()")
    action('EnableInput')
