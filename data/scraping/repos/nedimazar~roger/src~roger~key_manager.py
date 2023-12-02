from getpass import getpass
from cryptography.fernet import Fernet
import os
from pathlib import Path
import openai
import sys

# Get the directory of the current script
script_dir = Path(__file__).parent.absolute()


def load_key():
    with open(script_dir / "key.key", "rb") as key_file:
        return key_file.read()


def load_api_key():
    key = load_key()
    cipher_suite = Fernet(key)

    with open(script_dir / "api_key.key", "rb") as api_key_file:
        cipher_text = api_key_file.read()

    decrypted_text = cipher_suite.decrypt(cipher_text)
    return decrypted_text.decode()


def save_key(key):
    with open(script_dir / "key.key", "wb") as key_file:
        key_file.write(key)


def save_api_key(api_key):
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)

    cipher_text = cipher_suite.encrypt(api_key.encode())
    with open(script_dir / "api_key.key", "wb") as api_key_file:
        api_key_file.write(cipher_text)

    save_key(key)


def test_api_key(api_key):
    """Perform a simple API request to OpenAI to test the API key."""
    openai.api_key = api_key
    try:
        # Perform a simple API request
        openai.Engine.list()

        return True
    except openai.OpenAIError:
        return False


def get_api_key():
    if not os.path.exists(script_dir / "api_key.key"):
        api_key = getpass("Enter your OpenAI API key: ")
    else:
        api_key = load_api_key()

    if test_api_key(api_key):
        if not os.path.exists(script_dir / "api_key.key"):
            save_api_key(api_key)

        return api_key
    else:
        print("The provided OpenAI API key is not valid.")
        sys.exit(1)
