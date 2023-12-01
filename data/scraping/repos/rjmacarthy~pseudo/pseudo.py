import openai
import yaml
import sys
from rich.console import Console
from rich.markdown import Markdown

config = yaml.safe_load(open("config.yaml"))
model = config["model_engine"]
console = Console()


def get_completion(messages):
    return openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )


def start_sudo_session():
    sudo_lang = open("sudolang-llm-support/sudolang.sudo.md", "r").read()
    if not sudo_lang:
        console.print("Error: Could not load sudolang.")
        return
    messages = [{"role": "system", "content": sudo_lang}]
    return messages


def get_sudo_program(file_path):
    with open(file_path, "r") as file:
        return file.read()


def start(program_name):
    console.print(f"Loading {program_name} using pseudo version 1.0, Please wait...")
    load(program_name)
    

def load(program_name):
        messages = start_sudo_session()
        program = get_sudo_program(program_name)
        messages.append({"role": "user", "content": program})
        completion = get_completion(messages)
        console.print(f"Loaded program {program_name} successfully 🚀")
        reply = completion["choices"][0]["message"]["content"]
        console.print(Markdown(reply))
        while True:
            user_input = input("> ")
            messages.append({"role": "user", "content": user_input})
            completion = get_completion(messages)
            reply = completion["choices"][0]["message"]["content"]
            console.print(Markdown(reply))
            continue
            
def main():
    if len(sys.argv) < 2:
        console.print("Please enter a program name to run e.g `pseudo folder/program.sudo`")
        return
    start(sys.argv[1])
    