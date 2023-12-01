import os
import chainlit as cl
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

LLM_MODEL_NAME = "text-davinci-003"
SKILL_DIR = "../skills"
SKILL_COLLECTION = "FunSkill"

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAITextCompletion
kernel = sk.Kernel()
kernel.add_text_completion_service(
    service_id="dv",
    service=OpenAITextCompletion(
        LLM_MODEL_NAME,
        os.environ.get("OPENAI_API_KEY"),
        os.environ.get("OPENAI_ORG_ID")
    )
)
fun_skills = kernel.import_semantic_skill_from_directory(SKILL_DIR, SKILL_COLLECTION)
joke_skill = fun_skills.get("Joke")

@cl.on_message  
async def main(message: str):
    response = await joke_skill.invoke_async(message)
    await cl.Message(
        content=f"{response}"
    ).send()

@cl.on_chat_start
async def start():
    await cl.Message(
        content="Hello there!"
    ).send()
    
    
