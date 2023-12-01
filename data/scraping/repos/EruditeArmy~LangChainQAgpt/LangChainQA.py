# || OPENAI API KEY REQUIRED ||
# CREATE dot .env file as the same level as LangChainQA.py and put the key as OPENAI_API_KEY=key


#imports
import os
import sys

from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

#dot env for not leaking the openai API key
from dotenv import load_dotenv
load_dotenv()

#used for setting the openai api key to the model
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

#this is where you ask the model a question
# type this into the terminal to get a response : python3 LangChainQAgpt/LangChainQA.py "The question your asking"
#query = sys.argv[1]
query = "How did Ezra'a get her message out to others? a. She created an educational website b. She organized huge public protests c. She took a world tour and spoke countless countries d. She wrote letters to the politicians in Bahrain."

#loads a txt file where you can input data and the response will be based on the data
loader = TextLoader('LangChainQAgpt/langChainTEXTtest.txt')

#this can be used to load a directory of txt (used for multiple txt files)
#loader = DirectoryLoader(".", glob="*.txt")

index = VectorstoreIndexCreator().from_loaders([loader])

#prints response to terminal
print(index.query(query))