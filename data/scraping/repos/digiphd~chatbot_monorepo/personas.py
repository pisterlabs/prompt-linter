# app/personas.py
import openai  
import os
from tools.prompts import seo_blogging_expert_template, ceo_founder_template, youtube_creator_template
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate,  ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from loguru import logger
from common.config import OPENAI_API_KEY


memory = ConversationBufferWindowMemory(k=1)

def youtube_creator_chat(message: str) -> str:
    try:
        logger.debug(f'youtube logger:{message}')
        
        chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
        system_message_prompt = SystemMessagePromptTemplate.from_template(youtube_creator_template)
     
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        
        chat(
            chat_prompt.format_prompt(
                input_language="English", output_language="English", text=message
            ).to_messages()
        )
        
        conversation = ConversationChain(llm=chat, memory=memory)
        response = conversation.run(message)
        # Extract response or process LangChain response as needed
        # For simplicity, assuming LangChain response can be returned directly
        memory.chat_memory.add_user_message(message)
        memory.chat_memory.add_ai_message(response)
        
    except Exception as e:
        # Handle exceptions, perhaps logging them and returning a default message
        print(f"Error: {e}")
        response = "Sorry, there was an error processing your request."
    return response

def seo_blogging_expert_chat(message: str) -> str:
    return f"SEO and Blogging Expert says: {message}"

def ceo_founder_chat(message: str) -> str:
    return f"CEO and Founder says: {message}"


# TODO: Get memory working (single user) LLang chain memory - Done
# TODO: Get system prompt for each expert working - Thought I got this working, but doesn't seem to work correctly.
# TODO: Get longterm memory working with MongoDb
# TODO: Get user specific memory working - Perhaps start with telegram
# TODO: Think about possibly streaming the response back? Might be dependent on the platform. I.e. Telegram, Discord
# TODO: Get envvars working correctly