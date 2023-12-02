import os
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts import ChatMessagePromptTemplate
from langchain.prompts import MessagesPlaceholder
from constants import apiKey, serpapi_api_key


chat = ChatOpenAI(temperature=0)

template = "You are a helpful assistant that translates english to pirate."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
example_human = HumanMessagePromptTemplate.from_template(
    "Hi", additional_kwargs={"name": "example_user"})
example_ai = AIMessagePromptTemplate.from_template(
    "Argh me mateys", additional_kwargs={"name": "example_assistant"})
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        system_message_prompt,
        example_human,
        example_ai,
        human_message_prompt,
    ]
)

chain = LLMChain(llm=chat, prompt=chat_prompt)


def fewshot_cm():
    # get a chat completion from the formatted messages
    return chain.run("aphex twins")


# using MessagesPlaceholder
