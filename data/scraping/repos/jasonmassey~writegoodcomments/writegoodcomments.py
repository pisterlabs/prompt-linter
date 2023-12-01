#!/usr/bin/env python3

import os
import sys
import argparse
import getpass
import openai

# Constants
CONFIG_FILE_PATH = os.path.expanduser("~/.writegoodcomments")

# Set up OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Utility functions

def get_openai_api_key():
    api_key = input("Enter your OpenAI API key: ")
    with open(CONFIG_FILE_PATH, "w") as config_file:
        config_file.write(f"OPENAI_API_KEY={api_key}\n")

def read_config():
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, "r") as config_file:
            for line in config_file:
                key, value = line.strip().split("=")
                if key == "OPENAI_API_KEY":
                    return value
    return None

def get_file_extension(file_path):
    _, ext = os.path.splitext(file_path)
    return ext.lower()

def generate_comments(code, signature):
    # Use OpenAI API to generate comments
    response = openai.Completion.create(
        engine="davinci",
        prompt=code,
        max_tokens=100,
        temperature=0.7
    )
    return signature + response.choices[0].text

def process_file(file_path, signature):
    with open(file_path, "r") as file:
        code = file.read()

    comments = generate_comments(code, signature)

    with open(file_path, "w") as file:
        file.write(comments)

def main():
    parser = argparse.ArgumentParser(description="Generate detailed comments for code files using OpenAI.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recursively process code files.")
    parser.add_argument("-changesig", type=str, help="Change the comment signature.")
    parser.add_argument("files", nargs="*", help="List of code files to process.")

    args = parser.parse_args()

    if not args.files:
        parser.print_help()
        return

    openai_api_key = read_config()
    if openai_api_key is None:
        get_openai_api_key()

    signature = "j--"
    if args.changesig:
        signature = args.changesig

    for file_path in args.files:
        if os.path.isfile(file_path) and get_file_extension(file_path) in [".c", ".cpp", ".h", ".java", ".js"]:
            process_file(file_path, signature)

    if args.recursive:
        for root, _, files in os.walk("."):
            for file in files:
                file_path = os.path.join(root, file)
                if get_file_extension(file_path) in [".c", ".cpp", ".h", ".java", ".js"]:
                    process_file(file_path, signature)

if __name__ == "__main__":
    main()
