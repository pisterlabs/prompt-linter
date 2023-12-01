from langchain.llms import openai
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# from langchain.llms import OpenAI

llm = openai(model_name="text-davinci-003")
llm("explain large language models in one sentence")