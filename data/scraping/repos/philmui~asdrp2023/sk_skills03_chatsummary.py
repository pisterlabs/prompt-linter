import os
import chainlit as cl
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env"))
LLM_MODEL_NAME = "gpt-3.5-turbo"         # OpenAI

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = sk.Kernel()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_ORG_ID = os.environ["OPENAI_ORG_ID"]

# add text completion service 
kernel.add_chat_service(
    service_id="chat-gpt", 
    service=OpenAIChatCompletion(
        LLM_MODEL_NAME, 
        OPENAI_API_KEY, 
        OPENAI_ORG_ID
    )
)

SOLICITATION = "Type in some text for me to summarize!"

# key TODO : create a summarization prompt!

# create summarization skill
summarize_skill = # // TODO

@cl.on_message  
async def main(message: str):
    response = await summarize_skill.invoke_async(message)
    await cl.Message(
        content=f"{response}"
    ).send()

@cl.on_chat_start
async def start():
    await cl.Message(
        content=SOLICITATION
    ).send()
