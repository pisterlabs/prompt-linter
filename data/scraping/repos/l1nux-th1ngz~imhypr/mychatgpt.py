

import openai
import pyperclip
import yaml
import readline
import click

from pathlib import Path
from yaml import load,CLoader as Loader

# Path to your openai.yaml in your home directory
openaipath = "/private/openai.yaml"
# openaipath = "/mychatgpt/openai.yaml"
chat_logfile = "/private/chatlog.txt"

# Get home path
home = str(Path.home())

# Load openai api key
with open(home + openaipath, "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=Loader)

# Set up the OpenAI API client
openai.api_key = cfg["openai"]["api-key"]

# Set up the model and prompt
model_engine = "text-davinci-003"

@click.command()
def sendRequest():

    prompt = str(input("Hello, how can I help you? "))
    print('...')

    # Generate a response
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    response = completion.choices[0].text
    print(response)

    pyperclip.copy(response)
    print("...")
    print("Output has been copied to the clipboard!")
    
    c = input("Do you want to store the question and answer in the Chat Logfile (y/n)? (default: y) ")
    if (c != "n"):
        f = open (home + chat_logfile, "a")
        f.write("Question: " + prompt)
        f.write(response + "\n")
        f.write(" \n")
        f.close()
        print("Output written to chat logfile " + chat_logfile + "!")

if __name__ == '__main__':
    sendRequest()

