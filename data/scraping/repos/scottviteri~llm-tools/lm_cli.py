#!/usr/bin/env python

# Built-in modules
import sys
import json
import argparse
import subprocess
import os

# External module
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# This script acts as a command-line interface for interacting with
# the "Claude" AI model from the Anthropic API. It provides functionalities
# like loading and managing conversation histories, creating AI prompts,
# fetching AI completions, and executing shell commands based on AI's response.

def main(args):
    # In debug mode, print out all the arguments that were provided
    if args.debug:
        print(f"Chat IDs: {args.chat}")
        print(f"Input message: {args.message}")
        print(f"Read chat ID: {args.read}")
        print(f"Delete chat ID: {args.delete}")
        print(f"Undo chat ID: {args.undo}")
        print(f"Shell command: {args.shell}")
        print(f"Programming language: {args.proglang}")
        print(f"List flag: {args.list}")

    # Initialize the Anthropic API
    anthropic = Anthropic()

    # Load the conversation history. This is stored in a JSON file.
    history=''
    # lm.py
    # Get the directory of the current script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Specify the relative path to the claude_history.json file
    history_file = os.path.join(dir_path, 'claude_history.json')
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            histories = json.load(f)
    else:
        histories = {}
    # Prepare the history based on multiple chat IDs
    # If multiple chat IDs are provided, their histories are concatenated
    for chat_id in args.chat:
        if chat_id in histories:
            history += histories[chat_id]

    if args.debug:
        print(f"Prepared history: {history}")

    # If the '--list' option is provided, list all the chat IDs and exit
    if args.list:
        print("Chat IDs:")
        for chat_id in histories.keys():
            print(chat_id)
        return

    # If the '--read' option is provided, read the chat with the provided ID and exit
    if args.read:
        if args.read in histories:
            print(histories[args.read])
        else:
            print(f"No chat with ID {args.read} found.")
        return

    # If the '--undo' option is provided, undo the last message in the chat with the provided ID and exit
    if args.undo:
        if args.undo in histories:
            # split the chat history into rounds based on the HUMAN_PROMPT delimiter
            rounds = histories[args.undo].split(HUMAN_PROMPT)
            # remove the last round
            rounds = rounds[:-1]
            # join back the rounds into a single string
            histories[args.undo] = HUMAN_PROMPT.join(rounds)
            # save the updated history
            with open(history_file, 'w') as f:
                json.dump(histories, f)
            print(f"Last message in chat {args.undo} has been undone.")
        else:
            print(f"No chat with ID {args.undo} found.")
        return 

    # If the '--delete' option is provided, delete the chat with the provided ID and exit
    if args.delete:
        if args.delete in histories:
            histories.pop(args.delete, None)
            with open(history_file, 'w') as f:
                json.dump(histories, f)
            print(f"Chat with ID {args.delete} has been deleted.")
        else:
            print(f"No chat with ID {args.delete} found.")
        return 

    # Prepare the prompt for the AI. This is based on the provided message and the chosen command
    if args.shell:
        message = f"{args.message}. Use directly executable {args.shell} syntax, and if you need to think to yourself, put it in comments. If eshell, use elisp notation. The OS is arch linux."
    elif args.proglang:
        message = f"{args.message}. Please provide directly runnable code in the {args.proglang} programming language, without starting backticks or natural language text."
    else:
        message = args.message

    prompt = f"{history} {HUMAN_PROMPT} {message} {AI_PROMPT}"

    if args.debug:
        print(f"Prompt: {prompt}")

    # Get the completion from the AI. This is done by sending a request to the Anthropic API
    response = anthropic.completions.create(
        model="claude-2.0",
        max_tokens_to_sample=
        3000,
        prompt=prompt,
        stream=True
    )

    # Collect the response from the AI
    completion = ''
    for chunk in response:
        completion += chunk.completion
        print(chunk.completion, end='')

    if args.debug:
        print(f"Response: {completion}")

    # Update the conversation history with the new completion
    new_history = f"{history} {HUMAN_PROMPT} {message} {AI_PROMPT} {completion}"

    if args.debug:
        print(f"New history: {new_history}")

    # Save the updated history if a chat ID is provided
    if args.chat:
        # Check if the chat exists, if not create a new one
        if args.chat[-1] not in histories:
            histories[args.chat[-1]] = ""
        histories[args.chat[-1]] += f" {HUMAN_PROMPT} {args.message} {AI_PROMPT} {completion}"
        with open(history_file, 'w+') as f:
            json.dump(histories, f)


    ## If the shell command option is enabled, ask the user for confirmation before executing the command
    #if args.shell:
    #    confirm = input('\nDo you want to run this command? (y/n): ')
    #    if confirm.lower() == 'y':
    #        subprocess.run(completion, shell=True)

    print()

if __name__ == '__main__':
    # Initialize command-line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--message', type=str, help='Input message')
    parser.add_argument('-c', '--chat', nargs='*', default=[], type=str, help='Chat IDs')
    parser.add_argument('-r', '--read', type=str, help='Read chat by ID')
    parser.add_argument('-d', '--delete', type=str, help='Delete chat by ID')
    parser.add_argument('-u', '--undo', type=str, help='Undo last message in chat by ID')
    parser.add_argument('-s', '--shell', type=str, help='Request shell command')
    parser.add_argument('-p', '--proglang', type=str, help='Request program in language')
    parser.add_argument('-l', '--list', action='store_true', help='List all chat IDs')
    parser.add_argument('-db', '--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    main(args)

