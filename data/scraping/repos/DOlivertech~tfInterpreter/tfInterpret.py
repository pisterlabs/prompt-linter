#!/usr/bin/env python3

import openai
import argparse
import textwrap
import keyring
import sys

SERVICE_ID = 'tfinterpreter'
USER_ID = 'openai'

TOKEN_LIMIT = 8000


def get_api_key():
    api_key = keyring.get_password(SERVICE_ID, USER_ID)
    if api_key is None:
        print("No API key found. Please enter your OpenAI API key:")
        api_key = input()
        keyring.set_password(SERVICE_ID, USER_ID, api_key)
        print("Your API key has been securely stored.")
    return api_key


def set_api_key(api_key=None):
    api_key_stored = keyring.get_password(SERVICE_ID, USER_ID)
    if api_key_stored is not None:
        print("An API key is already set. Please clear the existing key first.")
        return
    if api_key is None:
        print("Please enter your OpenAI API key:")
        api_key = input()
    keyring.set_password(SERVICE_ID, USER_ID, api_key)
    print("Your API key has been securely stored.")


def clear_api_key():
    api_key = keyring.get_password(SERVICE_ID, USER_ID)
    if api_key is None:
        print("No API key is stored. Nothing to clear.")
        return
    keyring.delete_password(SERVICE_ID, USER_ID)
    print("Your API key has been deleted.")


class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        if 'unrecognized arguments' in message:
            sys.stderr.write('This argument is not supported.\n')
        if 'argument --set-key: expected one argument' in message:
            sys.stderr.write(
                'No key was provided. Please provide an API key. Take a look at the help file for available arguments\n')
        self.print_help()
        sys.exit(2)


parser = CustomArgumentParser(
    description='''This program interprets a Terraform plan output and translates it into simple terms. It splits the plan into chunks and provides a detailed yet easily understandable explanation of all the changes that will be made, highlighting any additions, deletions, or modifications.

Usage: python tfInterpret.py <path_to_your_file>

Replace <path_to_your_file> with the path to your Terraform plan output file. The program will read the file and provide an analysis of the Terraform plan.

For example: python tfInterpret.py /home/user/plan.txt

Use the --clear-key option to clear the stored OpenAI API key:

python tfInterpret.py --clear-key

Use the --set-key option to input a new OpenAI API key:

python tfInterpret.py --set-key <your_api_key>
If <your_api_key> is not provided, you will be prompted to enter it.
''')
parser.add_argument(
    'file', type=str, nargs='?', default=None, help='The path to the Terraform plan output file.')
parser.add_argument(
    '--clear-key', action='store_true', help='Clear the stored OpenAI API key.')
parser.add_argument(
    '--set-key', type=str, default='prompt', help='Input a new OpenAI API key. If no key is provided with this option, you will be prompted to enter it.')

args = parser.parse_args()

if args.clear_key:
    clear_api_key()
    sys.exit(0)

if args.set_key != 'prompt':
    set_api_key(args.set_key)
    sys.exit(0)

openai.api_key = get_api_key()


def read_terraform_plan(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data


def split_plan(plan):
    return textwrap.wrap(plan, TOKEN_LIMIT)


def is_relevant(chunk):
    return not chunk.isspace()


def interpret_plan_chunk(chunk, chunk_number, total_chunks):
    print(f"Analyzing chunk {chunk_number} of {total_chunks}...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant that translates Terraform plans into simple terms. You should provide a detailed yet easily understandable explanation of all the changes that will be made, highlighting any additions, deletions, or modifications. Avoid explaining what Terraform is doing or details about the '-out' option. Just state the facts and provide a brief analysis."},
        {"role": "user", "content": f"Please explain this part of the Terraform plan concisely and factually:\n{chunk}"},
        {"role": "user", "content": "What resources will be added, modified, or deleted? Provide a brief and factual explanation."},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        max_tokens=TOKEN_LIMIT,
    )

    result = f"Chunk {chunk_number} of {total_chunks}:\n{response['choices'][0]['message']['content'].strip()}\n\n---\nTokens used for this chunk: {response['usage']['total_tokens']}\n---"
    return result


def main():
    if args.file is None:
        print("Error: A file argument is required. Please provide the path to the Terraform plan output file.")
        return

    print("Reading and analyzing your plan output...")

    plan = read_terraform_plan(args.file)
    plan_chunks = split_plan(plan)

    print(f"The analysis has been split into {len(plan_chunks)} chunks.")

    interpretations = []
    for i, chunk in enumerate(plan_chunks):
        if is_relevant(chunk):
            interpretations.append(interpret_plan_chunk(
                chunk, i+1, len(plan_chunks)))
        else:
            print(
                f"Skipping chunk {i+1} of {len(plan_chunks)} as it does not contain relevant information.")

    print("\n".join(interpretations))


if __name__ == "__main__":
    main()
