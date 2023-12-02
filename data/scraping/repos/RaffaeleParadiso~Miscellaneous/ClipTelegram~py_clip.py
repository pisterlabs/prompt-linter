import ctypes
import os
import time

import openai
import requests
from dotenv import load_dotenv

load_dotenv()

CF_TEXT = 1

kernel32 = ctypes.windll.kernel32
kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
kernel32.GlobalLock.restype = ctypes.c_void_p
kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
user32 = ctypes.windll.user32
user32.GetClipboardData.restype = ctypes.c_void_p


def get_clipboard_text():
    user32.OpenClipboard(0)
    try:
        if user32.IsClipboardFormatAvailable(CF_TEXT):
            data = user32.GetClipboardData(CF_TEXT)
            data_locked = kernel32.GlobalLock(data)
            text = ctypes.c_char_p(data_locked)
            value = text.value
            kernel32.GlobalUnlock(data_locked)
            return value
    finally:
        user32.CloseClipboard()


def get_response(messages: list):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        temperature=1.0
    )
    return response.choices[0].message


if __name__ == "__main__":
    openai.api_key = os.environ["OpenAI"]
    messages = [{"role": "system",
                "content": "You are a virtual assistant with a very good understanding of ..."}]
    user_input_list = []
    try:
        while True:
            user_input = get_clipboard_text().decode('ANSI')
            if user_input is None:
                continue
            user_input_list.append(user_input)
            messages.append({"role": "user", "content": user_input})
            new_message = get_response(messages=messages)
            print(f"\nJOI: {new_message['content']}")
            messages.append(new_message)
            time.sleep(10)
            TOKEN = os.environ["TOKEN_BOT"]
            chat_id = os.environ["chat_id_TG"]
            url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={new_message['content']}"
            requests.get(url).json()
    except KeyboardInterrupt:
        with open("log.txt", "w") as f:
            f.write(str(user_input_list))
