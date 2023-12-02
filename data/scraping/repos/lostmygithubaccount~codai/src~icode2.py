# imports
import re
import os
import glob
import toml
import typer
import openai
import marvin
import requests

import logging as log

from rich import print
from rich.console import Console
from dotenv import load_dotenv

from marvin import ai_fn, ai_model
from pydantic import BaseModel, Field
from marvin.engine.language_models import chat_llm

## local imports
from src.common import codai

# load .env file
load_dotenv()

# load config.toml
config = {}
try:
    config = toml.load("config.toml")["icode2"]
except:
    pass

# variables
help_message = "yolo"
console = Console()

marvin.settings.llm_model = "azure_openai/gpt-4"


# icode
def icode_run():
    codai(end="")
    console.print(help_message)

    # history
    messages = []

    log.info("Starting icode2...")

    return
