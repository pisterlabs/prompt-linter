import os
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

key_value = os.getenv('API_KEY')
if key_value is None:
    raise Exception("API_KEY not found in environment variables")
os.environ["OPENAI_API_KEY"] = key_value

loader = DirectoryLoader(".", glob="*.txt")
index = VectorstoreIndexCreator().from_loaders([loader])

while True:
    query = input("What is your question (or type 'c' to close): ")
    if query == 'c':
        break
    print(index.query(query, llm=ChatOpenAI()))
