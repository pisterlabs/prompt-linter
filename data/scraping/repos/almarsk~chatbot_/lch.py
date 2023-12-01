from langchain.prompts import PromptTemplate
from langchain import OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ChatMessageHistory
import os
from utils import api_key
from prompt_memory_export import update_memory

async def fill_in(persona, task, user_id):
    api_key()
    messages = []
    update_memory(user_id, messages)
    chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

    messages.append(SystemMessage(content=persona))
    messages.append(HumanMessage(content=task))

    message = chat(messages)
    return message.content
