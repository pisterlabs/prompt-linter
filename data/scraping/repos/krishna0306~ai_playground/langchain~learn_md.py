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


MarkdownReader = download_loader("MarkdownReader")

loader = MarkdownReader()
documents = loader.load_data(file=Path('/Users/despiegk/code/github/despiegk/ai_playground/openai/readme.md'))

index = VectorStoreIndex.from_documents(documents)



query_engine = index.as_query_engine()
# query_engine.query('how vulnerable are security protocols?')

IPython.embed()
