import sys
import os
import openai
from flask import Flask

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

data_to_pass_back = "Send this to node process"

input = sys.argv[1]
output = data_to_pass_back
print(data_to_pass_back)

sys.stdout.flush()
