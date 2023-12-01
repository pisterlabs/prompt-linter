import os
import sys

from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

key_value = os.getenv('API_KEY')
if key_value is None:
    raise Exception("API_KEY not found in environment variables")
os.environ["OPENAI_API_KEY"] = key_value

query = sys.argv[1]
loader = DirectoryLoader(".", glob="*.txt")
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query, llm=ChatOpenAI()))
