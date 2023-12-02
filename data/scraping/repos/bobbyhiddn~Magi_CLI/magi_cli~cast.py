#!/usr/bin/env python3

import click # type: ignore
import subprocess
import sys
import os
import openai # type: ignore
import glob
from magi_cli.spells import commands_list, aliases

# Load the Openai API key
# This can also be done by setting the OPENAI_API_KEY environment variable manually.
# load_dotenv() # Uncomment this line if you want to load the API key from the .env file
api_key = os.getenv("OPENAI_API_KEY")

# Load the defauit .tome directory path
tome_path = os.getenv("TOME_PATH")

# Set the API key for the OpenAI package
openai.api_key = api_key

# Non-click functions

def execute_bash_file(filename):
    subprocess.run(["bash", filename], check=True)

# Updated execute_python_file function to accept args
def execute_python_file(filename, args):
    subprocess.run([sys.executable, filename, *args], check=True)

def execute_spell_file(spell_file):
    tome_path = os.getenv("TOME_PATH")
    spell_file_path = spell_file if '.tome' in spell_file else os.path.join(tome_path, spell_file)

    if not spell_file_path.endswith('.spell'):
        spell_file_path += '.spell'
    
    if not os.path.exists(spell_file_path):
        click.echo(f"Could not find {spell_file}.spell in .tome directory. Checking current directory for .tome directory...")
        spell_file_path = os.path.join(".tome", spell_file)
        if not os.path.exists(spell_file_path):
            click.echo(f"Could not find {spell_file}.spell in current directory or .tome directory.")
            return
        elif not spell_file_path.endswith('.spell'):
            spell_file_path += '.spell'

    with open(spell_file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("#"):
            continue
        subprocess.run(stripped_line, shell=True)

# Click functions

@click.command()
@click.argument('input', nargs=-1)
def cast(input):
    input = list(input)  # Convert input into a list to separate command and arguments

    tome_path = os.getenv("TOME_PATH", ".tome")

    if not input:
        # Display available commands and spells if no input is provided
        print("Available commands:")
        for name, command in cli.commands.items():
            print(f"- {name}: {command.help}")

        print("\nAvailable spells recorded in your tome:")
        for file in glob.glob(f"{tome_path}/*.spell"):
            print(f"- {os.path.basename(file)}")

        print("\nAvailable aliases:")
        for alias, command in aliases.items():
            print(f"- {alias}: {command.name}")

    elif input[0] in aliases:
        # If the input is a registered alias, invoke the corresponding command
        command = aliases[input[0]]
        ctx = click.get_current_context()
        ctx.invoke(command, file_paths=input[1:])

    elif input[0] in cli.commands:
        # If the input is a registered command, pass all other arguments to it
        ctx = click.get_current_context()
        ctx.invoke(cli.commands[input[0]], file_paths=input[1:])

    else:
        # Check if the input is a file and execute accordingly
        file_path = os.path.join(tome_path, input[0])
        if os.path.isfile(file_path) or os.path.isfile(input[0]):
            target_file = file_path if os.path.isfile(file_path) else input[0]
            if target_file.endswith(".py"):
                execute_python_file(target_file, input[1:])
            elif target_file.endswith(".spell"):
                execute_spell_file(target_file.replace(".spell", ""))
            elif target_file.endswith(".sh"):
                execute_bash_file(target_file)
        else:
            print(f"Error: Command or file '{input[0]}' not found.")


@click.group()
@click.pass_context
def cli(ctx):
    """A Python CLI for casting spells."""
    pass

for command in commands_list:
    cli.add_command(command)

if __name__ == "__main__":
    cast()
