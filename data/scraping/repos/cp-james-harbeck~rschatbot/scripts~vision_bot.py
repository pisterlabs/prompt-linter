import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import time
import pytesseract
import openai
from pynput.keyboard import Controller, Key
import random
import logging
import re
from typing import List, Tuple
import datetime
from collections import deque

# Set up OpenAI API credentials
openai.api_key = '<your-openai-api-key>'

# Set up Tesseract OCR path (replace with your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Define screen capture coordinates
SCREEN_WIDTH = 3896
SCREEN_HEIGHT = 2160
SCREEN_REGION = (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)

# Define chat input coordinates
CHAT_INPUT_X = 1000
CHAT_INPUT_Y = 1800

# Define the conversation history
conversation_history = deque(maxlen=3)  # Limit the conversation history to the three most recent messages

# Define keyboard controller
keyboard = Controller()

# Define hotkey for chat input
CHAT_HOTKEY = Key.enter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to simulate pressing and releasing a hotkey
def press_hotkey():
    try:
        keyboard.press(CHAT_HOTKEY)
        keyboard.release(CHAT_HOTKEY)
        time.sleep(1)
    except Exception as e:
        logger.error(f"An error occurred while pressing the hotkey: {str(e)}")

# Function to simulate typing a message character by character
def type_message(message):
    try:
        for char in message:
            keyboard.press(char)
            keyboard.release(char)
            time.sleep(0.1)
        keyboard.press(CHAT_HOTKEY)
        keyboard.release(CHAT_HOTKEY)
    except Exception as e:
        logger.error(f"An error occurred while typing the message: {str(e)}")

# Function to capture the screen using video capture
def capture_screen():
    try:
        # Calculate the screen capture region based on chat input coordinates
        x1 = max(0, CHAT_INPUT_X - 1000)
        y1 = max(0, CHAT_INPUT_Y - 1000)
        x2 = min(SCREEN_WIDTH, CHAT_INPUT_X + 1000)
        y2 = min(SCREEN_HEIGHT, CHAT_INPUT_Y + 2000)
        
        # Capture the screen region
        screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))
        captured_region = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        return captured_region
    except Exception as e:
        logger.error(f"An error occurred while capturing the screen: {str(e)}")
        return None

# Function to extract text from an image using Tesseract OCR
def extract_text(image):
    try:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Perform preprocessing on the grayscale image if needed (e.g., thresholding, denoising, etc.)
        # Extract text using Tesseract OCR
        text = pytesseract.image_to_string(grayscale)
        return text
    except Exception as e:
        logger.error(f"An error occurred while extracting text from image: {str(e)}")
        return ''

# Function to send a message to the game
def send_message(message):
    try:
        # Split the message into words
        words = message.split()
        num_words = len(words)

        if num_words > 15:
            # The message is too long, truncate it
            truncated_message = ' '.join(words[:15])
            logger.info(f"Truncated message to 15 words: {truncated_message}")
            # Switch focus to the RuneScape window
            window = gw.getWindowsWithTitle('RuneScape')[0]
            window.activate()
            time.sleep(0.5)

            # Press the hotkey to open chat input
            press_hotkey()
            time.sleep(0.5)

            # Type the truncated message
            type_message(truncated_message)
            time.sleep(0.5)
        else:
            # The message fits within the limit, send it normally
            logger.info(f"Sending message: {message}")
            # Switch focus to the RuneScape window
            window = gw.getWindowsWithTitle('RuneScape')[0]
            window.activate()
            time.sleep(0.5)

            # Press the hotkey to open chat input
            press_hotkey()
            time.sleep(0.5)

            # Type the message
            type_message(message)
            time.sleep(0.5)
    except Exception as e:
        logger.error(f"An error occurred while sending the message: {str(e)}")

# Function to process player messages
def process_player_messages(text: str) -> List[Tuple[str, str]]:
    lines = text.split('\n')
    player_messages = []
    pattern = re.compile(r'^(\w+):\s*(.*)$')  # Assumes username contains only word characters
    for line in lines:
        match = pattern.match(line)
        if match:
            name = match.group(1)
            message = match.group(2)
            player_messages.append((name, message))
    return player_messages  # Reverse the order of player messages

# Function to generate a response using GPT-3.5-turbo
def generate_message(user_prompt, system_prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use GPT-3.5-turbo
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
            max_tokens=15,  # Adjust as per your requirements
        )
        
        # Get the generated response
        bot_response = response['choices'][0]['message']['content'].strip()

        # Truncate the response if it exceeds 80 characters
        truncated_response = bot_response[:70]
        
        return truncated_response
    except Exception as e:
        logger.error(f"An error occurred while generating the response: {str(e)}")
        return None

# Main bot loop
def run_bot():
    last_message_time = time.time()
    is_bot_typing = False  # Track whether the bot is currently typing a response
    while True:
        try:
            # Capture the screen and extract text
            screenshot = capture_screen()

            if screenshot is not None:
                # Show the captured screen
                cv2.imshow('Screen', screenshot)
                cv2.waitKey(1)  # Required for OpenCV to update the window

                text = extract_text(screenshot)
                logger.info(f"Extracted text: {text}")

                # Process player messages
                player_messages = process_player_messages(text)

                if player_messages:
                    # Select the last player message
                    name, message = player_messages[-1]

                    if name == "GEwhisperer":
                        continue  # Skip processing the message

                    # Combine the message history and current message as context
                    context = '\n'.join([f"{name}: {message}" for name, message in player_messages])
                    user_prompt = f"{context}\n{name}: {message}"
                    system_prompt = "You're a regular RuneScape player, you make super short funny comments in conversation in under 70 characters. Never ever mention that you are an AI model, or an AI at all, no exceptions! If you can't find a relevant response, ask a question to ignite conversation. Without exception, respond in under 70 characters. Without exception, always respond in a full sentence. Do not use hashtags or proper grammar. Do not use emojis. Less is more here. Here is the message from the user that you will respond to:"

                    if is_bot_typing:
                        # Wait for the bot to finish typing before generating a new response
                        time.sleep(2)
                        is_bot_typing = False

                    response = generate_message(user_prompt, system_prompt)

                    if response:
                        logger.info(f"Player message: {name}: {message}")  # Log the player message

                        # Delay the bot's response to simulate typing
                        typing_delay = random.uniform(1, 2)
                        logger.info(f"Typing response: {response}")
                        logger.info(f"Bot is typing for {typing_delay:.2f} seconds")
                        is_bot_typing = True

                        # Send the response as a message in the game
                        send_message(response)
                        # Update the last message time
                        last_message_time = time.time()

                # Clear the conversation history
                player_messages.clear()

            # Sleep for a random duration
            sleep_time = random.uniform(10, 10)
            elapsed_time = time.time() - last_message_time
            if elapsed_time < sleep_time:
                logger.info(f"Sleeping for {sleep_time - elapsed_time:.2f} seconds")
                time.sleep(sleep_time - elapsed_time)

        except KeyboardInterrupt:
            logger.info("Program terminated by user")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            time.sleep(10)  # Wait a bit before the next iteration to prevent rapid-fire error messages

        # Sleep after capturing the screen to prevent rapid screenshotting
        time.sleep(1)

    # Release the keyboard
    keyboard.release(CHAT_HOTKEY)
    # Close the OpenCV windows
    cv2.destroyAllWindows()

# Call the main function to start the bot
run_bot()