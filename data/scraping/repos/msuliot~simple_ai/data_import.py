from pathlib import Path
from llama_index import download_loader
import openai

# get keys from .env file
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

PDFReader = download_loader("PDFReader") # https://llamahub.ai

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

loader = PDFReader()
documents = SimpleDirectoryReader('data').load_data()
index = GPTVectorStoreIndex.from_documents(documents)
index.storage_context.persist()
print("Done") 