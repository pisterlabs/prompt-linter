import os
import chainlit as cl
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env"))
LLM_MODEL_NAME = "text-davinci-003"         # OpenAI
SKILL_DIR        = "../skills"
SKILL_COLLECTION = "FunSkill"

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAITextCompletion

kernel = sk.Kernel()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_ORG_ID = os.environ["OPENAI_ORG_ID"]

# add text completion service 
# // TODO

SOLICITATION = "What do you want a joke to be about?"

@cl.on_message  
async def main(message: str):
    response = # // TODO
    await cl.Message(
        content=f"{response}"
    ).send()

@cl.on_chat_start
async def start():
    await cl.Message(
        content=SOLICITATION
    ).send()
