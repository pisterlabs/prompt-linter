from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate

chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

system_message_prompt = SystemMessagePromptTemplate.from_template("You are a helpful assistant that translates english into the pirate.")
example_human = HumanMessagePromptTemplate.from_template("Hi.")
example_ai = AIMessagePromptTemplate.from_template("Argh me mateys!")
human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai, human_message_prompt])
chain = LLMChain(llm=chat, prompt=chat_prompt)
result = chain.run("I love programming.")
print(result)