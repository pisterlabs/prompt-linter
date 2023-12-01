
import requests
import os
import openai
import io
from elevenlabs import generate, play, voices, set_api_key
import pyperclip
import pygetwindow as gw
import win32process
from pyautogui import typewrite, hotkey


def process_text(predicted_text, ctx):
    if not keyb_is_permitted():
        return
    save = pyperclip.paste()
    pyperclip.copy(predicted_text)
    hotkey("ctrl", "v")
    pyperclip.copy(save)
    #     return
    # typewrite(predicted_text)


def keyb_is_permitted():

    active_window = gw.getActiveWindow()

    # Access the HWND (Handle to Window) of the active window
    hwnd = active_window._hWnd

    # Find the process ID associated with the HWND
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    # if 'Visual Studio' in active_window.title:
        # return False 
    return True

    return predicted_text


