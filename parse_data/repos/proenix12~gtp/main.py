import json
import os
import time
import sys
from pygments import highlight
from pygments.lexers import BashLexer
from pygments.formatters import TerminalFormatter

# install pakages if not installed
# pip install openai
# pip install pygments


from openai import OpenAI


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    argument = sys.argv[1]

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        organization=os.environ.get("OPENAI_ORGANIZATION_KEY")
    )

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "user",
                "content": argument
            }
        ],
        max_tokens=255,
        temperature=0.7
    )

    for choice in response.choices:
        highlighted_code = highlight(choice.message.content, BashLexer(), TerminalFormatter())
        print("Role: ", highlighted_code)

    print("-----------------------------------")

    print("Log:", response.model, "Usage: Completion Tokens:", response.usage.completion_tokens,
          "Prompt Tokens:", response.usage.prompt_tokens, "Total Tokens:", response.usage.total_tokens)

    # log to file
    log_file_path = os.path.join(os.path.dirname(__file__), "log.txt")

    with open(log_file_path, "a") as myfile:
        for choice in response.choices:
            myfile.write(choice.message.content + '\n')
