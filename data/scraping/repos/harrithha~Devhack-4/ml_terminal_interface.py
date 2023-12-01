import os
import re
import sys
import openai
import colorama
import subprocess
from colorama import Fore

openai.api_key = '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'

messages = [
        {"role": "system", "content": "You are a helpful assistant."},
]
message = "pretend you are a bot who gives bash commands for the instructions that I give. If the instruction that I give are valid, which can be done using command line, you should only give the valid command without any descriptions or quotation marks. If the commands are not possibible give following text as output 'ERROR'"
messages.append(
                {"role": "user", "content": message},
        )
chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
)


def run_command(command):
    """
    Runs a bash command and returns the output.
    Throws an error if the command is not valid.
    """
    result = subprocess.run(command, stdout=subprocess.PIPE, shell=True, check=None, stderr=subprocess.PIPE)

    if result.returncode != 0:
        return result.stderr.decode().strip(), 1
    
    return result.stdout.decode().strip(), 0

def translate_to_command(command: str):
    """
    Translating human reaadable command to bash command.
    This uses Large Language Model. (GPT)
    """
    message = command
    message = "Give only the bash command to " + message + "without any other descriptive text"
    if message:
        messages.append(
                {"role": "user", "content": message},
        )
        chat_completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
        )
    answer = chat_completion.choices[0].message.content
    nw = answer.split("```")[0]
    messages.append({"role": "assistant", "content": nw})
    output = f"{nw}"
    new_output = re.sub(r"(^`|`$)", "", output)
    return new_output


    