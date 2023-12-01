import time
import random
import sys
import openai
import pyautogui
from pynput.keyboard import Controller, Key
import datetime
import logging
import configparser
from collections import deque
import threading

# Read configuration file
config = configparser.ConfigParser()
try:
    config.read('config.ini')
except configparser.Error as e:
    logging.error(f"An error occurred while reading the configuration file: {str(e)}")
    sys.exit(1)

# Setup logging
logging.basicConfig(filename='runescape_bot.log', level=logging.INFO)

# Variables from configuration file
try:
    hotkey = Key[config.get('Settings', 'hotkey')]
    window_title = config.get('Settings', 'window_title')
    min_sleep_time = config.getfloat('Settings', 'min_sleep_time')
    max_sleep_time = config.getfloat('Settings', 'max_sleep_time')
    min_typing_speed = config.getfloat('Settings', 'min_typing_speed')
    max_typing_speed = config.getfloat('Settings', 'max_typing_speed')
    openai.api_key = config.get('Settings', 'openai_api_key')
    max_tokens = config.getint('Settings', 'max_tokens')
    model = config.get('Settings', 'model')
    prompt_1 = config.get('user_prompts', 'prompt_1')
    prompt_2 = config.get('user_prompts', 'prompt_2')
    system_prompt_morning = config.get('system_prompts', 'morning_prompt')
    system_prompt_afternoon = config.get('system_prompts', 'afternoon_prompt')
    system_prompt_night = config.get('system_prompts', 'night_prompt')
except configparser.Error as e:
    logging.error(f"An error occurred while reading a setting from the configuration file: {str(e)}")
    sys.exit(1)

# Effects for messages
effects = [
    'red:shake:', 'red:slide:', 'red:wave:', 'red:wave2:',
    'green:shake:', 'green:slide:', 'green:wave:', 'green:wave2:',
    'cyan:shake:', 'cyan:slide:', 'cyan:wave:', 'cyan:wave2:',
    'purple:shake:', 'purple:slide:', 'purple:wave:', 'purple:wave2:',
    'white:shake:', 'white:slide:', 'white:wave:', 'white:wave2:',
    'flash1:wave:', 'flash2:wave:', 'flash3:wave:', 
    'glow1:wave:', 'glow2:wave:', 'glow3:wave:'
]

keyboard = Controller()
stop_event = threading.Event()

def press_hotkey():
    """
    Simulates pressing and releasing a hotkey
    """
    try:
        keyboard.press(hotkey)
        keyboard.release(hotkey)
        time.sleep(1)
    except Exception as e:
        logging.error(f"An error occurred while pressing the hotkey: {str(e)}")

def type_message(message):
    """
    Types out a message character by character with simulated delay between keypresses
    """
    try:
        for char in message:
            if stop_event.is_set():  # Check if the stop event has been signaled
                logging.info("Stopping bot in the middle of typing.")
                break
            keyboard.press(char)
            keyboard.release(char)
            time.sleep(random.uniform(min_typing_speed, max_typing_speed))
        keyboard.press(Key.enter)
        keyboard.release(Key.enter)
    except Exception as e:
        logging.error(f"An error occurred while typing the message: {str(e)}")

def get_runescape_windows():
    """
    Returns a list of all RuneScape windows
    """
    try:
        all_windows = pyautogui.getAllWindows() # get all windows
        windows = deque([w for w in all_windows if w.title.lower() in ['runescape', 'old school runescape']]) # filter windows
        if not windows:
            raise Exception("No RuneScape window found")
        return windows
    except Exception as e:
        logging.error(f"An error occurred while trying to focus on the window: {str(e)}")
        raise e

def generate_message(user_prompt, system_prompt):
    """
    Generates a message based on the current UTC time using GPT-3
    """
    logging.info(f"Current UTC time: {datetime.datetime.utcnow().isoformat()}")
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            max_tokens=max_tokens
        )
        message = response['choices'][0]['message']['content']
        if message is None:
            message = "Join the FriendchatRS FC!"
        message = message.strip('"')
        effect = random.choice(effects)
        message = f"{effect} {message}"
        message = message.encode('ascii', 'ignore').decode('ascii')

        logging.info(f"Generated message: {message}")
        return message
    except Exception as e:
        logging.error(f"An error occurred while generating the message: {str(e)}")
        return None

def start_bot(bot_config):
    stop_event.clear()
    try:
        windows = get_runescape_windows()
    except Exception as e:
        logging.error(f"An error occurred while getting the RuneScape windows: {str(e)}")
        sys.exit(1)

    while not stop_event.is_set():
        try:
            sleep_time = random.uniform(min_sleep_time, max_sleep_time)
            logging.info(f"Sleeping for {sleep_time/60:.2f} minutes")

            start_time = time.time()
            while time.time() - start_time < sleep_time:
                if stop_event.is_set():
                    return
                time.sleep(1)

            if windows:
                window = windows.popleft()  # Get the next window
                windows.append(window)  # Add the window back to the end of the queue

                window.activate()
                press_hotkey()

                if stop_event.is_set():
                    return

            current_time = datetime.datetime.utcnow()
            time_interval = int(bot_config['Settings']['time_interval'])
            user_prompts = list(bot_config['user_prompts'].values())
            prompt_index = (current_time.minute // time_interval) % len(user_prompts)
            user_prompt = user_prompts[prompt_index]

            current_hour = datetime.datetime.utcnow().hour
            if 6 <= current_hour < 12:
                system_prompt = bot_config['system_prompts']['morning_prompt']
            elif 12 <= current_hour < 18:
                system_prompt = bot_config['system_prompts']['afternoon_prompt']
            else:
                system_prompt = bot_config['system_prompts']['night_prompt']

            message = generate_message(user_prompt, system_prompt)
            if message is not None:
                logging.info(f"Typing message: {message}")
                type_message(message)

            if stop_event.is_set():
                return

            start_time = time.time()
            while time.time() - start_time < sleep_time:
                if stop_event.is_set():
                    return
                time.sleep(1)

        except KeyboardInterrupt:
            logging.info("Program terminated by user")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")
            time.sleep(60)  # Wait a bit before the next iteration to prevent rapid-fire error messages

def stop_bot():
    stop_event.set()