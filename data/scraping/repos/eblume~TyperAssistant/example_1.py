"""Usage: python example_1.py ask --no-prompt "Please greet me by name"
"""
import getpass

import typer
from openai import OpenAI
from typerassistant import TyperAssistant

app = typer.Typer()
client = OpenAI()  # Assuming OPENAI_API_KEY is set in the environment


@app.command()
def say_hello(name: str):
    print(f"Hello, {name}!")


@app.command()
def get_current_user():
    print(getpass.getuser())


TyperAssistant(app, client=client, replace=True)
app()
