import os
import sys

import constants
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = constants.APIKEY

query = sys.argv[1]
print(query)

# loader = TextLoader("data.txt")
loader = DirectoryLoader("data/")
index = VectorstoreIndexCreator().from_loaders([loader])

# printing the answer from your own data only
# print(index.query(query))

# printing the data from both your data and the AI model
print(index.query(query, llm=ChatOpenAI()))
