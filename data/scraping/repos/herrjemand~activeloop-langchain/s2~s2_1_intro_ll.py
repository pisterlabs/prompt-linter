from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

system_prompt = SystemMessagePromptTemplate.from_template(template = "You are an assistant that helps users find information about movies")
human_prompt = HumanMessagePromptTemplate.from_template(template = "Find the information about the movie {movie_title}")
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
response = chat(chat_prompt.format_prompt(movie_title="The Matrix").to_messages())

print(response.content)