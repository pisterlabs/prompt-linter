# Import necessary libraries
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain import hub

# Additional imports
import os
import re
import time

# Initialize the OpenAI client (ensure your API key is set up)
client = langchain.llms.OpenAI()

# Setup the directory for database
persist_directory = 'db'
embedding = OpenAIEmbeddings()

# Initialize the vector database
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
from chromaviz import visualize_collection

visualize_collection(vectordb._collection)

# git clone https://github.com/mtybadger/chromaviz.git
# cd chromaviz
# pip3 install .
# python3 visualize.py