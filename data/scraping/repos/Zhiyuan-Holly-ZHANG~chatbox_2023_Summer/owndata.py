import os 
import sys
import constants
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

os.environ["OPEN_API_KEY"] = constants.APIKEY

query = sys.argv[1]

 
loader = TextLoader('data1.txt')
# print("hello")
# loader = DirectoryLoader(".",glob="*.txt")

index = VectorstoreIndexCreator().from_loaders([loader])
# print(index.query(query))

print(index.query(query,llm=ChatOpenAI()))

# 科学园在哪