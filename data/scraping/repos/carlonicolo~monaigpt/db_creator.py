from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import api_key as key
import os

os.environ['OPENAI_API_KEY'] = key.OPENAI_API_KEY

loader = DirectoryLoader('./data')
pages = loader.load_and_split()

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embedding=embeddings, persist_directory="./monai_gpt_db")
vectordb.persist()