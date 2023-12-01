import os
import sys

import envVar
#from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import openai
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = envVar.APIKEY

#from langchain.chat_models import ChatOpenAI
#llm = ChatOpenAI(openai_api_key="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")

query = sys.argv[1]
print(query)

#loader = TextLoader('Data.txt')

loader = DirectoryLoader("./alldatahere", glob="*.txt")

index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query), llm=ChatOpenAI())