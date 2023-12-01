#!/usr/bin/env python3

# This is a chat bot like CLI aims for getting answers from chat in a loop
# Updates will include any models for it (TBD)

import os
from dotenv import load_dotenv, find_dotenv
import logging
import sys

from openai import OpenAI

load_dotenv(find_dotenv())
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

log_format = "[%(levelname)s][%(asctime)s][%(funcName)s] %(message)s"
logging.basicConfig(level=logging.WARNING, format=log_format)
logger = logging.getLogger(__name__)


VERSION = "0.1.3"
# Make this configurable
GPT_MODEL_VERSION = "gpt-4"


def main():
    print(f"AI Agent v{VERSION}, model version: {GPT_MODEL_VERSION}")
    print("Type help for more commands in the console")

    system_message = {"role": "system", "content": "You are a intelligent assistant."}
    messages = [system_message]
    while True:
        try:
            # TODO: handle the input with newline
            # Currently when there's newline pasted in the console
            # the process will executed automatically, which is
            # not ideal. Should distingush between the enter and newline
            # to fix this.
            message = input("> ")
            strip_message = message.strip().lower()
            # Ensure it is not an empty string
            if strip_message:
                if strip_message == "clear":
                    messages = [system_message]
                elif strip_message == "help":
                    print(
                        "command: clear for starting a new chat without history (context)"
                    )
                else:
                    messages.append(
                        {"role": "user", "content": message},
                    )
                    chat = client.chat.completions.create(
                        model=GPT_MODEL_VERSION, messages=messages
                    )
                    reply = chat.choices[0].message.content
                    print(f"AI: {reply}\n")
                    messages.append({"role": "assistant", "content": reply})
        except KeyboardInterrupt:
            messages = [system_message]
            print("KeyboardInterrupt, clear all context")
        except EOFError:
            sys.exit(0)


if __name__ == "__main__":
    main()
