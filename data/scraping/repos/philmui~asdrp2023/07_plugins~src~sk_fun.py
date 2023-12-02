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
kernel.add_text_completion_service(
    service_id="dv", 
    service=OpenAITextCompletion(
        LLM_MODEL_NAME, 
        OPENAI_API_KEY, 
        OPENAI_ORG_ID
    )
)
skill = kernel.import_semantic_skill_from_directory(SKILL_DIR, SKILL_COLLECTION)
joke_skill = skill.get("Joke")
excuse_skill = skill.get("Excuses")
poem_skill = skill.get("Limerick")

SOLICITATION = "Tell me a subject about a joke, an excuse, or a poem!"

def route_message(message: str):
    if "joke" in message.lower():
        return joke_skill(message)
    elif "excuse" in message.lower():
        return excuse_skill(message)
    elif "poem" in message.lower():
        return poem_skill(message)
    else:
        return SOLICITATION
    
@cl.on_message  
async def main(message: str):
    response = route_message(message)
    await cl.Message(
        content=f"{response}"
    ).send()

@cl.on_chat_start
async def start():
    await cl.Message(
        content=SOLICITATION
    ).send()
