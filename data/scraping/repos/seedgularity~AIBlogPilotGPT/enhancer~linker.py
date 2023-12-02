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
    fast_llm_model = cfg.fast_llm_model
except KeyError:
    sys.stderr.write("OpenAI key configuration failed.")
    exit(1)

def add_links(article):
    # Split the article into chunks of approximately 100 words each
    word_list = article.split()
    chunks = [' '.join(word_list[i:i + 100]) for i in range(0, len(word_list), 100)]

    chunked_output = ""
    for chunk in chunks:
        prompt = f"""You are an expert in SEO and blogging. A blogger wrote the following article.
            Your task is the review the article and add links to relevant sources. Add 5 links maximum.
            Article:{chunk}"""

        for output in openai.ChatCompletion.create(
                model=llm_model,
                stream=True,
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'system', 'content': "You're an expert in blogging, research and SEO."},
                    {'role': 'system', 'content': 'Your name is BloggingGPT.'},
                    {'role': 'system', 'content': 'Content you write is well SEO Optimised.'},
                    {'role': 'system', 'content': 'You use engaging tone of voice.'},
                    {"role": "user", "content": prompt}
                ]
        ):
            content = output["choices"][0].get("delta", {}).get("content")
            if content is not None:
                #print(content, end='')
                chunked_output += content

    return chunked_output