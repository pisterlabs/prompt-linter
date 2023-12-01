from colorama import Fore, Style
import os
import signal
import subprocess
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
import readline
import getch

import openai

# loading env
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# loading options
f = open('options.json')
options = json.load(f)

openai.api_key = os.getenv('OPENAI_API_KEY')

MODEL = "davinci"

# get open ai command
def get_command(prompt, option, stop):
    results = openai.Answer.create(
        search_model=MODEL,
        model=MODEL,
        question=prompt,
        examples_context=option['examples_context'],
        examples=option['examples'],
        max_tokens=100,
        documents=[],
        stop=stop,
    )
    if results:
        return results['answers'][0]

# format home path
def get_current_path():
    return os.getcwd().replace(os.path.expanduser('~'), '~')

# shell texts and style
def default_text():
    return f"{Fore.CYAN}[{get_current_path()}]{Fore.GREEN}{Style.BRIGHT}$ {Style.RESET_ALL}"


def machine_text(text):
    return f"{Fore.GREEN}{Style.BRIGHT}{MODEL}: {Fore.MAGENTA}{Style.NORMAL}{text}{Style.RESET_ALL}"

# handle the cd command
def handle_cd(request):
    if request.startswith("cd "):
        os.chdir(request[3:])
        return True

    if request.strip() == "cd":
        os.chdir(os.path.expanduser('~'))
        return True

    return False

# handle the ai
def handle_ai(request):
    option = 'bash'
    stop = ["\n", "<|endoftext|>"]
    if request.startswith('python:'):
        option = 'python'
        stop = ["<|endoftext|>"]
    elif request.startswith('node:'):
        option = 'node'
        stop = ["<|endoftext|>"]

    print(machine_text("🧠 Thinking..."))

    new_command = get_command(request, options[option], stop)

    if not new_command:
        print(machine_text("Unable to figure out how to do that"))
        return
    
    print(machine_text(new_command))

    if not option == 'bash':
        return
    
    print(default_text() + new_command)
    key_stroke = getch.getch()

    if key_stroke == '\n':
        os.system(new_command)

# shell running and stuff
def main():
    print(machine_text("Hello."))

    while True:
        try:
            request = input(default_text())
        except EOFError:
            print("")
            print(machine_text("Farewell, human."))
            sys.exit(0)
        except KeyboardInterrupt:
            print("")
            continue

        if not request.strip():
            continue

        if request.strip() == "exit":
            sys.exit(machine_text("Farewell, human."))
            continue

        if handle_cd(request):
            continue

        if request.startswith('->'):
            handle_ai(request[2:])
            # do ai stuff in here
            continue

        os.system(request)


if __name__ == "__main__":
    main()
