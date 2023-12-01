from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
#import magic
import os
import nltk

import unstructured
#import python-magic-bin
import chromadb

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

nltk.download('averaged_perceptron_tagger')

#pip install unstructured
# Other dependencies to install https://langchain.readthedocs.io/en/latest/modules/document_loaders/examples/unstructured_file.html
#pip install python-magic-bin
#pip install chromadb

loader = DirectoryLoader('/data', glob='**/*.txt')

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
docsearch = Chroma.from_documents(texts, embeddings)
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])


query = "quando deve ser pedida a primeira ecografia?"
print(qa.run(query))

