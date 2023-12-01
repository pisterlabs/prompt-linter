#!/usr/bin/env python3

from gpt_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from dotenv import load_dotenv
import sys
import os
import pathlib


# load environment variables from .env file
load_dotenv()

# get the value of the OPENAI_API_KEY environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def ask_ChatAfrica(query):
    index_path = pathlib.Path(__file__).parent / "index.json"
    index = GPTSimpleVectorIndex.load_from_disk(str(index_path))
    response = index.query(query, response_mode="compact", verbose=False)
    return response.response
#     print (response.response)

# if __name__ == "__main__":
#     ask_ChatAfrica(query=sys.argv[1])
