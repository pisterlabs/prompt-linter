#!/usr/bin/env python3
# Jussi.py
from dotenv import load_dotenv
import os
import openai
import sys
import html
import select
import tiktoken
import threading
import time
import hashlib
import re
import argparse
from typing import List
from PyPDF2 import PdfReader

# Pre-compile regular expressions
HTML_TAG_RE = re.compile(r"<.*?>")
URL_RE = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)

# Function to clean text based on various parameters
def clean_text(
    input_text: str,
    remove_html: bool = True,
    remove_urls: bool = True,
    replace_tabs_spaces: bool = True,
    replace_newlines: bool = True,
) -> str:
    """Clean the input text based on the given parameters."""
    if remove_html:
        input_text = HTML_TAG_RE.sub("", input_text)
    if remove_urls:
        input_text = URL_RE.sub("", input_text)
    if replace_tabs_spaces:
        input_text = re.sub(r"\s+", " ", input_text)
    if replace_newlines:
        input_text = input_text.replace("\n", " ")
    return input_text


# Function to parse arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean text based on various parameters."
    )
    parser.add_argument("--remove_html", action="store_true", help="Remove HTML tags")
    parser.add_argument("--remove_urls", action="store_true", help="Remove URLs")
    parser.add_argument(
        "--replace_tabs_spaces",
        action="store_true",
        help="Replace tabs and extra spaces with a single space",
    )
    parser.add_argument(
        "--replace_newlines",
        action="store_true",
        help="Replace newline characters with spaces",
    )
    parser.add_argument("--import-file", type=str, help="Path to the PDF file to be imported")
    parser.add_argument("extra_args", nargs="*", help="Extra arguments")
    return parser.parse_args()


# Function to calculate the number of tokens in a text string
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# Function to create the ~/.jussiai directory if it doesn't exist
def create_jussiai_directory():
    jussiai_dir = os.path.expanduser("~/.jussiai")
    if not os.path.exists(jussiai_dir):
        os.makedirs(jussiai_dir)


# Function to create or overwrite the .env file with the provided API key in the ~/.jussiai directory
def create_env_file(api_key):
    env_path = os.path.expanduser("~/.jussiai/.env")
    with open(env_path, "w") as env_file:
        env_file.write(f"OPENAI_API_KEY={api_key}")


# Function to load the .env file from the primary path or the current directory
def load_env_file():
    default_path = os.path.expanduser("~/.jussiai/.env")
    if os.path.exists(default_path):
        load_dotenv(dotenv_path=default_path)
        return True
    elif os.path.exists(".env"):
        load_dotenv()
        return True
    else:
        return False


# Function to sanitize user input to include only allowed characters and limit the length
def sanitize_input(input_text, max_length):
    allowed_chars = set(
        ".:-()+abcdefghijklmnopqrstuvwxyzåäö ABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ"
    )
    sanitized_input = "".join(char for char in input_text if char in allowed_chars)[
        :max_length
    ]
    sanitized_input = html.escape(sanitized_input)
    return sanitized_input


# Function to make an API call to OpenAI
def api_call(messages):
    global completion
    try:
        # Set the OpenAI API key
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Create a new sha256 hash object
        hash_object = hashlib.sha256()

        # Encode the username and update the hash object with the bytes-like object
        hash_object.update(os.getlogin().encode())

        # Get the hexadecimal representation of the hash
        username_hash = hash_object.hexdigest()

        # Make a request to the OpenAI API
        completion = openai.ChatCompletion.create(
            model="gpt-4", messages=messages, temperature=0, user=username_hash
        )
    except openai.error.APIError as e:
        # Handle API error here, e.g. retry or log
        print(f"\nOpenAI API returned an API Error: {e}")
        pass
    except openai.error.APIConnectionError as e:
        # Handle connection error here
        print(f"\nFailed to connect to OpenAI API: {e}")
        pass
    except openai.error.RateLimitError as e:
        # Handle rate limit error (we recommend using exponential backoff)
        print(f"\nOpenAI API request exceeded rate limit: {e}")
        pass


# Function to display a spinning cursor while waiting
def spinning_cursor():
    while True:
        for cursor in "|/-\\":
            yield cursor


# Main function
def main():
    create_jussiai_directory()

    # Attempt to load the .env file
    if not load_env_file():
        # If .env file is not found, prompt the user for the OpenAI API key
        api_key = input("Please provide your OpenAI API key: ")
        create_env_file(api_key)
        print(
            "An .env file with your API key has been created in the ~/.jussiai directory."
        )

    # Check if the user provided text as a parameter
    if len(sys.argv) < 2:
        print("Please provide text as a parameter!")
        return

    # Parse the arguments
    args = parse_arguments()

    # Read from stdin if available
    i, o, e = select.select([sys.stdin], [], [], 0.1)
    if i:
        file_content = sys.stdin.read()
    else:
        file_content = ""

    # Sanitize and concatenate extra arguments
    max_length = 100
    user_input = " ".join(args.extra_args) if args.extra_args else ""
    sanitized_input = sanitize_input(user_input, max_length)

    # Clean the text based on the arguments
    file_content = clean_text(
        file_content,
        args.remove_html,
        args.remove_urls,
        args.replace_tabs_spaces,
        args.replace_newlines,
    )

    if args.import_file:
        with open(args.import_file, 'rb') as file:
            reader = PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                file_content += page.extract_text()

    # Calculate the number of tokens in the sanitized input and file content
    num_tokens = num_tokens_from_string(
        sanitized_input + " " + file_content, "cl100k_base"
    )

    # Check if the token limit is exceeded
    if num_tokens > 40000:
        print(
            f"Token limit exceeded ({num_tokens}). You are allowed to use up to 40,000 tokens per minute."
        )
        return

    # Define the initial messages for the chat
    messages = [
        {
            "role": "system",
            "content": "You are a helpful software developer.",
        },
        {"role": "user", "content": sanitized_input + " " + file_content},
    ]

    # Start the API call in a separate thread
    thread = threading.Thread(target=api_call, args=(messages,))
    thread.start()

    # Display a spinning cursor while waiting
    spinner = spinning_cursor()
    sys.stdout.write("Processing your request... ")
    sys.stdout.flush()
    while thread.is_alive():
        sys.stdout.write(next(spinner))
        sys.stdout.flush()
        time.sleep(0.1)
        sys.stdout.write("\b")
        sys.stdout.flush()

    # Extract and print the assistant's response
    response = completion["choices"][0]["message"]["content"]

    print("\n" + response)


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
