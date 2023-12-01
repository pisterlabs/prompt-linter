from dive.util.configAPIKey import set_openai_api_key
from examples.base import index_example_data,query_example_data,clear_example_data
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI
import time

#Use chromadb and OpenAI embeddings and llm


set_openai_api_key()
index_example_data(256, 20, False, OpenAIEmbeddings(), OpenAI())
print('------------Finish Indexing Data-----------------')
time.sleep(30)
print('------------Start Querying Data-----------------')
question='What is airbnb\'s revenue?'
instruction = None
#instruction = 'answer this question in Indonesian only'
openai_model='gpt-3.5-turbo'
query_example_data(question, 4, OpenAIEmbeddings(), OpenAI(temperature=0), instruction,openai_model)
#clear_example_data()