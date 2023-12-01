import os
import sys
import time
import openai
import keyboard
from dotenv import load_dotenv


def loading_message(num_dots):
    for i in range(num_dots):
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(0.5)

    sys.stdout.write("\r")
    sys.stdout.flush()


def get_user_input():
    user_input = input("")

    if (
        user_input.lower() == "/exit"
        or user_input.lower() == "/quit"
        or user_input.lower() == "/q"
    ):
        return None

    user_input = "message: " + user_input
    return user_input


def process_conversation(user_input, history):
    messages = []

    for input_text, completion_text in history:
        messages.append({"role": "user", "content": input_text})
        messages.append({"role": "assistant", "content": completion_text})

    messages.append({"role": "user", "content": user_input})

    loading_message(3)

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=200,
            temperature=0.7,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            top_p=1,
        )

    except Exception as e:
        print(f"Error: {e}")
        return None

    sys.stdout.write("\r" + " " * (len("connecting...") + 3) + "\r")
    sys.stdout.flush()

    completion_text = completion.choices[0].message.content
    return completion_text


load_dotenv()

openai.organization = "org-kr1OG3zTxuHnLhRvWEusigNZ"
openai.api_key = os.getenv("OPENAI_API_KEY")

history = []
firstRun = True
args = sys.argv
inputString = ""

if len(args) > 1:
    for arg in args:
        if firstRun == True:
            firstRun = False
            continue
        inputString += arg + " "

preamble = "Please respond to all following user messages as if you were an edgy hacker from an 80s movie. Ensure answers are less than 200 tokens. \n\n"

if inputString == "":
    user_input = preamble + "Good day hacker..."
else:
    user_input = preamble + inputString

while len(history) < 5:
    if keyboard.is_pressed("esc"):
        break

    if len(history) == 0:
        completion_text = process_conversation(user_input, history)
    else:
        user_input = get_user_input()

        if user_input is None:
            break

        completion_text = process_conversation(user_input, history)

    if completion_text is None:
        break

    print(f"{completion_text}")

    history.append((user_input, completion_text))

time.sleep(1)

print("dropping connection...")
