from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

messages = [
    SystemMessage(content="You are a helpful assistant that translates sentences from English to French"),
    HumanMessage(content="Translate: Hello, how are you?"),
]

print(chat(messages))


batch_messages = [
  [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate the following sentence: I love programming.")
  ],
  [
    SystemMessage(content="You are a helpful assistant that translates French to English."),
    HumanMessage(content="Translate the following sentence: J'aime la programmation.")
  ],
]

print( chat.generate(batch_messages) )