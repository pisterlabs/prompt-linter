# OpenAI Variables: Contains variables for OpenAI Models and API

import os

import openai
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

OPENAI_API_KEY = os.getenv('API_KEY') or 'API_KEY'
openai.api_key = OPENAI_API_KEY

llm_model = "gpt-4"  # OpenAI LLM engine model
temp = 0.5  # OpenAI LLM temperature

embed_model = 'text-embedding-ada-002'  # OpenAI Embedding model
embed = OpenAIEmbeddings(model = embed_model, openai_api_key = OPENAI_API_KEY)  # Embedding variable

llm = ChatOpenAI(openai_api_key = OPENAI_API_KEY, model_name = llm_model, temperature = temp)