#!/usr/bin/env python3

import pyautogui
import pyperclip
import time
import os
from pynput import mouse
import platform
import openai

def on_click(x, y, button, pressed):
    if button == mouse.Button.middle and pressed:
        handle_text()

def handle_text():
    pltfm = platform.platform()
    if "Windows" in pltfm:
        ctrl = 'ctrl'
    elif "macOS" in pltfm:
        ctrl = 'command'
    else:
        print("UNSUPPORTED PLATFORM")
        ctrl = 'ctrl'

    time.sleep(0.4)
    # Step 1: Get the currently selected text
    pyautogui.hotkey(ctrl, 'c', interval=0.1)  # Copy selected text to clipboard
    time.sleep(0.1)  # Wait for the clipboard to be updated
    selected_text = pyperclip.paste()

    # Step 2: Delete the selected text
    pyautogui.press('delete')
    time.sleep(0.1)
    # Step 3: Select all text
    pyautogui.hotkey(ctrl, 'a', interval=0.1)
    time.sleep(0.1)

    # Step 4: Save all selected text into a variable
    pyautogui.hotkey(ctrl, 'c', interval=0.1)
    time.sleep(0.1)
    all_text = pyperclip.paste()

    # Step 5: Go to the start of the document
    pyautogui.hotkey(ctrl, 'up', interval=0.1)
    time.sleep(0.1)


    promt = f"""
    Always be polite, professional, not too formal and always try to match the tone of the mails from the sender.
    You will write as Simon.
    The Content you should write about is: {selected_text}
    The History of the conversation is as follows: {all_text}
    Create a reply. Do not write a "Subject" line, directly write the mail.
    """

    # Step 6: Get the generated text from openAI
    messages = [
        {"role": "system", "content": "You are my assistant who will write Mails in my name, which is Simon"},
        {"role": "user", "content": promt},
    ]
    stream = True
    resp = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    stream=stream,
    )
    if stream:
        content = []
        for delta_resp in resp:
            finished = delta_resp['choices'][0]['finish_reason'] is not None
            if "content" in  delta_resp['choices'][0]['delta']:
                word = delta_resp['choices'][0]['delta']['content']
                content.append(word)
                print(word, end="", flush=True)
            pass
        print()
        content = ''.join(content)

    pyperclip.copy(content)
    pyautogui.hotkey(ctrl, 'v', interval=0.1)
    time.sleep(0.1)
    pyautogui.press('enter', interval=0.1)

    time.sleep(0.1)

def main():
    #prep openai
    openai.organization = os.getenv("OPENAI_ORG")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Start listening for mouse events
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()

if __name__ == "__main__":
    main()
