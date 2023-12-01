import time
import subprocess
from board import SCL, SDA, D4
import busio
import digitalio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1305
import curses
import sys
import openai
from collections import deque

# Initialize the OLED display
oled_reset = digitalio.DigitalInOut(D4)
i2c = busio.I2C(SCL, SDA)
disp = adafruit_ssd1305.SSD1305_I2C(128, 32, i2c, reset=oled_reset)
width = disp.width
height = disp.height

# Drawing objects
image = Image.new('1', (width, height))
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

# Set your OpenAI API key here
openai.api_key = 'your-api-key'

# deque with maxlen will automatically remove old items when the limit is reached
history = deque(maxlen=2000)
history.append(">")

# Scrolling variables
scroll = 0
max_scroll = 0

# Function for ChatGPT API


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    print(response.choices[0].message["content"])
    return response.choices[0].message["content"]


def draw_text():
    global scroll
    global max_scroll

    draw.rectangle((0, 0, width, height), outline=0, fill=0)

    # Draw up to 4 lines of history
    for i in range(4):
        if scroll + i < len(history):
            draw.text((0, 8*i), history[scroll + i], font=font, fill=255)

    disp.image(image)
    disp.show()


def wrap_text(text, width):
    # Split the text by spaces to get words
    words = text.split(' ')
    lines = []
    line = ''

    for word in words:
        if len(line + ' ' + word) <= width:
            line += ' ' + word
        else:
            lines.append(line.strip())  # Strip leading and trailing spaces
            line = word

    # Add any leftover text
    if line != '':
        lines.append(line.strip())  # Strip leading and trailing spaces

    return lines


def terminal(stdscr):
    global history
    global scroll
    global max_scroll

    input_str = ""
    input_lines = 0
    draw_text()

    while True:
        c = stdscr.getch()

        # ENTER
        if c == 10:
            if input_str.strip() != "":
                resp = get_completion(input_str)

                # Note the start of the response
                response_start = len(history) - input_lines

                # Add response to history, with line wrapping
                for line in wrap_text("A>" + resp, 20):
                    history.append(line)

                input_str = ""
                input_lines = 0
                # Reduced by one for less lines
                max_scroll = len(history) - 3 if len(history) > 3 else 0

                # Point scroll to the start of the response
                scroll = response_start if response_start <= max_scroll else max_scroll
        # BACKSPACE
        elif c == 256 or c == curses.KEY_BACKSPACE:
            input_str = input_str[:-1]
        # UP ARROW
        elif c == 259:
            if scroll > 0:
                scroll -= 1
        # DOWN ARROW
        elif c == 258:
            if scroll < max_scroll:
                scroll += 1
        else:
            if 32 <= c <= 126:  # Check if c is a printable ASCII character
                input_str += chr(c)

        # Remove the old input lines from the history
        for _ in range(input_lines):
            if len(history) > 0:
                history.pop()

        # Add user input to history, with line wrapping
        if input_str.strip() != "":
            # check if the only character in history is >
            if len(history) == 1 and history[0] == ">":
                history.pop()
            wrapped_input = list(wrap_text(">" + input_str, 20))
            input_lines = len(wrapped_input)
            for line in wrapped_input:
                history.append(line)

        draw_text()


if __name__ == "__main__":
    curses.wrapper(terminal)
