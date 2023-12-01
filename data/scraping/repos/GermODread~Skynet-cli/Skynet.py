#!/usr/bin/python3

import openai
import os
import sys

### Add the following line to .bashrc or .zshrc or any profile
# export OPENAI_API_KEY='API key'

api = 'OPEN_AI_API'  # Not recommended to keep it here. Save using OpenAI builtin asset management function.
cmd = sys.argv[1]
args = sys.argv[2:]
question = " ".join(args)


def helperdocumentation():
    print("Available functions:")
    print(skynetstream.__doc__)
    print(editgpt.__doc__)
    print(imgcreate.__doc__)


def skynetstream(q, z):
    """Chat completion with GPT-4
    Not using stream version"""
    openai.api_key = os.getenv(z)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Welcome to the new World"},
            {"role": "user", "content": f"{q}"},
            {"role": "assistant", "content": ""}
            ],
    )
    print(response['choices'][0]['message']['content'])


def imgcreate(q, z):
    """Image creation.
    Use "img" followed by text description of image, to create an image using Dall-E".
    Uses dall-e language model to create images based on text.
    """
    openai.api_key = os.getenv(z)
    response = openai.Image.create(prompt=f"{q}", n=1, size="1024x1024")

    print(response.data)


def editgpt(q, z):
    """Text proofing.
    Use "edit" followed by text to proof any spelling mistakes.
    Uses text-davinci-edit-001 to correct any spelling mistakes.
    """
    openai.api_key = os.getenv(z)
    response = openai.Edit.create(
        model="text-davinci-edit-001",
        input=f"{q}",
        instruction="Fix the spelling mistakes",
    )
    print(response.choices[0].text)


if __name__ == "__main__":
    if cmd == "stream":
        skynetstream(question, api)
    if cmd == "img":
        imgcreate(question, api)
    if cmd == "edit":
        editgpt(question, api)
    if cmd == "help":
        helperdocumentation()
