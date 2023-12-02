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
from llama_hub.tools.google_search.base import GoogleSearchToolSpec
from llama_index.agent import OpenAIAgent



tool_spec = GoogleSearchToolSpec(key="AIzaSyBymYdbUAYQ0oO66C8hUNZ9N_cj3G5SbcE", engine="47c5fbc1550aa447c")
pprint(tool_spec.google_search("weather today"))
agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())

agent.chat("Please create an event on aug 13th, 2023 at 5pm for 1 hour and invite xyz@abc.com to discuss tax laws")
r=agent.chat('What is on my calendar for today?')
pprint(r)


# index = VectorStoreIndex.from_documents([document])

# query_engine = index.as_query_engine()
# query_engine.query('how vulnerable are security protocols?')

IPython.embed()
