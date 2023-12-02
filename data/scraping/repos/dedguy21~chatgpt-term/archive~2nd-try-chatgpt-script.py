#! /usr/bin/env python
import os
import sys
import threading
import time
import readline
import rlcompleter  # Import the rlcompleter module
import openai
from colorama import Fore, Style

#model_id = "gpt-3.5-turbo-16k"
model_id = "gpt-4"

# Enable readline for tab completion
readline.parse_and_bind("tab: complete")


# Set the autocomplete function
def complete_path(text, state):
    """Auto-complete the file path."""
    return [path for path in os.listdir() if path.startswith(text)][state]


# Create a custom Completer class
class Completer(rlcompleter.Completer):
    def complete(self, text, state):
        if text.startswith(
            "/"
        ):  # If the input starts with "/", use the file path completer
            return complete_path(text[1:], state)
        else:  # Use the default completer for other cases
            return super().complete(text, state)


# Set the completion function for readline using the custom Completer class
readline.set_completer(Completer().complete)


def chatGPT_conversation(conversation):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(model=model_id, messages=conversation)
    conversation.append(
        {
            "role": response.choices[0].message.role,
            "content": response.choices[0].message.content,
        }
    )
    return conversation


def read_file_content(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return ""


def hourglass_animation():
    symbols = ["⌛", "⏳"]
    while not stop_animation_event.is_set():
        for symbol in symbols:
            sys.stdout.write(f"\r{symbol} Awaiting response...")
            sys.stdout.flush()
            time.sleep(0.5)


conversation = []
conversation.append({"role": "system", "content": "How may I help you?"})
conversation = chatGPT_conversation(conversation)
print(
    "{0}: {1}\n".format(
        conversation[-1]["role"].strip(), conversation[-1]["content"].strip()
    )
)

while True:
    action = input(
        "Do you want to (1) Enter multiline input or (2) Read from a file? (Enter 1 or 2): "
    )

    if action == "1":
        prompt = ""
        print(
            f"{Fore.RED}Enter your query{Style.RESET_ALL} (Press Enter then 'Ctrl + D' to submit): "
        )
        try:
            while True:
                line = input()
                prompt += line + "\n"
        except EOFError:
            pass
        except KeyboardInterrupt:
            print(f"{Fore.YELLOW}Exiting program.{Style.RESET_ALL}")
            sys.exit()
    elif action == "2":
        # Enable the completer for file paths
        readline.set_completer(complete_path)
        file_path = input("Enter the path to the file: ")
        prompt = read_file_content(file_path)
        # Disable the completer after reading the file
        readline.set_completer(None)
        if (
            not prompt
        ):  # If the prompt is empty (due to file error), continue to the next iteration
            continue
    else:
        print("Invalid choice. Please enter 1 or 2.")
        continue

    conversation.append({"role": "user", "content": prompt})

    # Start the hourglass animation
    stop_animation_event = threading.Event()
    animation_thread = threading.Thread(target=hourglass_animation)
    animation_thread.start()

    conversation = chatGPT_conversation(conversation)

    # Stop the hourglass animation
    stop_animation_event.set()
    animation_thread.join()

    print(f"{Fore.YELLOW}OpenAI response:{Style.RESET_ALL}")
    print(
        "{0}: {1}\n".format(
            f"{Fore.LIGHTBLUE_EX}{conversation[-1]['role'].strip()}{Style.RESET_ALL}",
            f"{Fore.CYAN + Style.BRIGHT}{conversation[-1]['content'].strip()}{Style.RESET_ALL}",
        )
    )
