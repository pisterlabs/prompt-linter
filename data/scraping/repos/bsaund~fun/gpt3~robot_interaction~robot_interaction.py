import openai
from gpt3.key import get_secret_key
from colorama import Fore, Style
from gpt3.utils import query_most_likely
import random

openai.api_key = get_secret_key()

FILENAME = "../stored_contexts/past_robot_interactions.txt"
PROMPT_SEPARATOR = " => I will"

ALLOWED_COMMANDS = {'move', 'pick', 'place'}


def load():
    with open(FILENAME) as f:
        return f.read()


def save(context):
    with open(FILENAME, 'w') as f:
        f.write(context)


def add_to_context(new_human_query, new_desired_command):
    context = load()
    context += new_human_query + PROMPT_SEPARATOR + " " + new_desired_command + "\n"
    save(context)


def add_context(query):
    context = load()
    return context + query + PROMPT_SEPARATOR


def handle_bad_completion(query):
    print("Which should I have done?")
    selected = input(f"{ALLOWED_COMMANDS}  ")
    if selected not in ALLOWED_COMMANDS:
        print(f"{Fore.RED}{selected}{Style.RESET_ALL} is not one of the allowed commands")
        return
    add_to_context(query, selected)


def single_interaction():
    query = input("What would you like me to do?   ")
    respose = query_most_likely(add_context(query), allowed_set=ALLOWED_COMMANDS)
    print(f"I will {Fore.GREEN}{respose}{Style.RESET_ALL}")
    if input("Did I interpret this correctly? (Y/n)") == 'n':
        handle_bad_completion(query)

def mainloop():
    load()
    while True:
        single_interaction()


mainloop()
