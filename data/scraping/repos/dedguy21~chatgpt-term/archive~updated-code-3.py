#! /usr/bin/env python
import os
import sys
import threading
import time
import pyperclip
import openai
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from pyfzf import FzfPrompt

console = Console()
fzf = FzfPrompt()

# Prompt user to select a file
def select_file():
    # Set the base_path to the user's home folder
    base_path = os.path.expanduser("~")

    # Recursively search for files in all directories
    all_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            all_files.append(os.path.join(root, file))

    # Use fzf to select one of the files
    selected_files = fzf.prompt(all_files, "--preview 'ls -l {}' --preview-window up:50%:wrap")

    if not selected_files:
        console.print("No file selected.", style="bold red")
        return ""

    file_path = selected_files[0]

    console.print("Selected file:")
    console.print(Panel(file_path, title="File Path", border_style="bold cyan"))

    # Copy selected file path to clipboard
    pyperclip.copy(file_path)
    console.print("The file path has been copied to the clipboard.", style="bold cyan")

    # Print the selected file path and provide an opportunity for edits
    while True:
        file_path_edit = input("Edit file path if necessary, or press enter to confirm: ").strip()
        if file_path_edit:
            file_path = file_path_edit
            break
        elif pyperclip.paste():
            file_path = pyperclip.paste().strip()
            console.print(f"Using clipboard file path: {file_path}\n")
            break
        else:
            break

    return file_path

# Generate chat response using OpenAI API
def chatGPT_conversation(model_id, conversation):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(model=model_id, messages=conversation)
    conversation.append(
        {
            "role": response.choices[0].message.role,
            "content": response.choices[0].message.content,
        }
    )
    return conversation

# Display hourglass animation
def hourglass_animation():
    console.print("Searching for pattern in all directories...")
    symbols = ["⌛", "⏳"]
    while not stop_animation_event.is_set():
        for symbol in symbols:
            sys.stdout.write(f"\r{symbol} Awaiting response...")
            sys.stdout.flush()
            time.sleep(0.5)

# Ask which model to use
while True:
    try:
        model_id = int(input("Enter the model to use (1) 'gpt-3.5-turbo-16k' or (2) 'gpt-4': "))
        if model_id == 1:
            model_id = "gpt-3.5-turbo-16k"
            break
        elif model_id == 2:
            model_id = "gpt-4"
            break
        else:
            console.print("Invalid choice. Please enter 1 or 2.", style="bold red")
    except ValueError:
        console.print("Invalid choice. Please enter a number.", style="bold red")

conversation = []
conversation.append({"role": "system", "content": "How may I help you?"})

# Generate initial response from the chosen model
conversation = chatGPT_conversation(model_id, conversation)

# Print the initial response
console.print(
    "{0}: {1}\n".format(
        conversation[-1]["role"].strip(), conversation[-1]["content"].strip()
    ),
    style="bold cyan",
)

# Main interaction loop
while True:
    # Prompt the user to choose an action
    while True:
        try:
            action = int(
                input(
                    "Do you want to (1) Enter multiline input, (2) Select a file or (3) Quit? "
                )
            )
            if action < 1 or action > 3:
                console.print("Invalid choice. Please enter 1, 2, or 3.", style="bold red")
            else:
                break
        except ValueError:
            console.print("Invalid choice. Please enter a number.", style="bold red")

    if action == 1:
        prompt = ""
        console.print(f"Enter your query (Press Enter then 'Ctrl + D'): ", style="bold red")
        try:
            while True:
                line = input()
                prompt += line + "\n"
        except EOFError:
            pass

    elif action == 2:
        console.print("Select a file using fzf. ", style="bold red")
        file_path = select_file()
        prompt = ""

        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                prompt = file.read()
            if prompt == "":
                console.print("File content is empty.", style="bold red")
        elif file_path != "":
            console.print(f"Error: File {file_path} not found.", style="bold red")
            continue

    elif action == 3:
        # Quit the program
        console.print("Quitting program...", style="bold red")
        break

    # Append user's input to the conversation list
    conversation.append({"role": "user", "content": prompt})

    # Initialize the hourglass animation
    stop_animation_event = threading.Event()
    animation_thread = threading.Thread(target=hourglass_animation)
    animation_thread.start()

    # Generate response from the OpenAI model
    conversation = chatGPT_conversation(model_id, conversation)

    # Stop the hourglass animation
    stop_animation_event.set()
    animation_thread.join()

    # Print the OpenAI response
    console.print("OpenAI response:", style="bold yellow")
    console.print(Panel(Markdown(f"{conversation[-1]['content'].strip()}\n"), title="OpenAI Response", border_style="bold cyan"))
