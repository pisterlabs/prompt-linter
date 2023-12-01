from llama_index import SimpleDirectoryReader, VectorStoreIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
from pprint import pprint; import IPython
import sys
import os
from pathlib import Path

# Check if the environment variable exists
if "OPENAIKEY" in os.environ:
    # If it exists, get its value into a Python variable
    api_key = os.environ["OPENAIKEY"]
else:
    raise ValueError("Please set the OPENAIKEY environment variable")
os.environ["OPENAI_API_KEY"] = api_key  

from llama_index import VectorStoreIndex, download_loader
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader('/Users/despiegk1/Downloads/ai').load_data()
index = GPTVectorStoreIndex.from_documents(documents)
index.storage_context.persist()

query_engine = index.as_query_engine()
query_engine.query("what is ourworld?")

# ImageReader = download_loader("ImageReader")
# imageLoader = ImageReader(text_type="plain_text")
# FlatPdfReader = download_loader("FlatPdfReader")
# pdfLoader = FlatPdfReader(image_loader=imageLoader)


# document = pdfLoader.load_data(file=Path('~/Downloads/its not about what we have, its about what we believe in. (5).pdf'))

# index = VectorStoreIndex.from_documents([document])

# query_engine = index.as_query_engine()
# query_engine.query('how vulnerable are security protocols?')

IPython.embed()
