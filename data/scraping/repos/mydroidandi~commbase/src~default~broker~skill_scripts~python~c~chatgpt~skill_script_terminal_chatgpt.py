#!/usr/bin/env python
################################################################################
#                                   Commbase                                   #
#                                                                              #
# Conversational AI Assistant and AI Hub for Computers and Droids              #
#                                                                              #
# Change History                                                               #
# 05/05/2023  Esteban Herrera Original code.                                   #
#                           Add new history entries as needed.                 #
#                                                                              #
#                                                                              #
################################################################################
################################################################################
################################################################################
#                                                                              #
#  Copyright (c) 2022-present Esteban Herrera C.                               #
#  stv.herrera@gmail.com                                                       #
#                                                                              #
#  This program is free software; you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation; either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program; if not, write to the Free Software                 #
#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA   #

# terminal_chatgpt.py
# A terminal version of ChatGPT
# ChatGPT is a chatbot built using the GPT (Generative Pre-trained Transformer)
# architecture developed by OpenAI.
# You can access ChatGPT by going to https://chat.openai.com/

# Imports
import openai
import typer
import rich
from rich import print
from rich.table import Table
import os


def get_chatgpt_api_key(callback=None):
    # Specify the path of the env file containing the variable
    file_path = os.environ["COMMBASE_APP_DIR"] + "/config/app.conf"

    # Open the file and read its contents
    with open(file_path, "r") as f:
        for line in f:
            # Split the line into variable name and value
            variable_name, value = line.strip().split("=")

            # Check if the variable we are looking for exists in the line
            if variable_name == "OPENAI_API_KEY":
                # Remove the quotes from the value of the variable
                API_KEY = value.strip()[1:-1]

                # Call the callback function with the API key value as an
                # argument
                if callback is not None:
                    callback(API_KEY)

                    return API_KEY

    # If the variable is not found, return None
    return None


def main():
    def process_api_key(api_key):
        # Do something with the API key value
        # print(f"Received API key: {API_KEY}")
        openai.api_key = api_key

    # Get the API key value and pass it to the callback function
    get_chatgpt_api_key(callback=process_api_key)

    print("[bold green]Terminal ChatGP[/bold green]")

    table = Table("Command", "Description")
    table.add_row("exit", "Exit the application")
    table.add_row("new", "New conversation")

    print(table)

    # Assistant context
    context = {"role": "system", "content": "You are a very helpful assistant."}
    messages = [context]

    while True:

        content = __prompt()

        if content == "new":
            print("ChatGPT: New conversation created")
            messages = [context]
            content = __prompt()

        messages.append({"role": "user", "content": content})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )

        response_content = response.choices[0].message.content

        messages.append({"role": "assistant", "content": response_content})

        print(f"[bold green]ChatGPT: [/bold green] [green]{response_content}[/green]")


def __prompt() -> str:
    prompt = typer.prompt("\nYou")

    if prompt == "exit":
        exit = typer.confirm("Are you sure?")
        if exit:
            print("Goodbye!")
            raise typer.Abort()

        return __prompt()

    return prompt


if __name__ == "__main__":
    typer.run(main)
