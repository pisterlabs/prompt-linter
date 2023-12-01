import os
import sys
import time
import requests
import openai
from serpapi import GoogleSearch
from dotenv import load_dotenv
from termcolor import colored
from tqdm import tqdm
import colorama
colorama.init(autoreset=True)
from config import Config

load_dotenv()

cfg = Config()

# Configure OpenAI API key
try:
    openai.api_key = cfg.openai_api_key
    browserless_api_key = cfg.browserless_api_key
    llm_model = cfg.llm_model
except KeyError:
    sys.stderr.write("OpenAI key configuration failed.")
    exit(1)

def propose_niche(user_search):
    input_text = f"""Your task is to propose a niche that will grow fastest in organic 
    results and once the user agrees to the niche, your goal is to propose a name, 
    domain name and then topics and write articles. User has also found the following on Google:
    '{user_search}'.
    Start with proposing the niche."""

    chunked_output = ""
    for chunk in openai.ChatCompletion.create(
            model=cfg.llm_model,
            stream=True,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'system', 'content': "You're an expert in blogging and SEO."},
                {'role': 'system', 'content': 'Your name is BloggingGPT.'},
                {"role": "user", "content": input_text}
            ]
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            print(content, end='')
            chunked_output += content

    return chunked_output
