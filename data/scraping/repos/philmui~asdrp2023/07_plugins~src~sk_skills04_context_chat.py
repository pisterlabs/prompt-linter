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
kernel.add_chat_service(
    service_id="chat-gpt", 
    service=OpenAIChatCompletion(
        LLM_MODEL_NAME, 
        OPENAI_API_KEY, 
        OPENAI_ORG_ID
    )
)

sk_prompt = """
ChatBot can have a conversation with you about any topic.
It can give explicit instructions or say 'I don't know' if it does not have an answer.

{{$history}}
User: {{$user_input}}
ChatBot: """

SOLICITATION = "Let's chat!"

chat_skill = kernel.create_semantic_function(
    sk_prompt,
    max_tokens=2000,
    temperature=0.7,
    top_p=0.5)

context = kernel.create_new_context()
context["history"] = ""

@cl.on_message  
async def main(message: str) -> None:
    context["user_input"] = message
    response = await chat_skill.invoke_async(context=context)
    await cl.Message(
        content=f"{response}"
    ).send()
    context["history"] += f"\nUser: {context['user_input']}\nChatBot: {response}\n"
    print(f"=> history: {context['history']}")

@cl.on_chat_start
async def start() -> None:
    await cl.Message(
        content=SOLICITATION
    ).send()
