import openai
import os
import sys
# import tiktoken

# !{sys.executable} pip3 install python-dotenv #, tiktoken

from dotenv import load_dotenv, find_dotenv  # Corrected module name
_ = load_dotenv(find_dotenv())  # Read local .env file

openai.api_key = os.getenv('OPENAI_API_KEY')

def getOpenAi():
  return openai
