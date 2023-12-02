import os
import sys
import json
from dotenv import load_dotenv
import openai
import toml
from halo import Halo
import colorama
from termcolor import colored

from config import Config

# Load the .env file
load_dotenv()

# Configure OpenAI API key
cfg = Config()
try:
    openai.api_key = cfg.openai_api_key
except KeyError:
    sys.stderr.write("OpenAI key configuration failed.")
    exit(1)


def init_blog(description, published_articles, toneofvoice, keywords):
    if keywords is None:
        prompt = """I have the following blog with the following description: """ + description + """.
        The blog has published articles with titles: """ + str(published_articles) + """.
        The blog has following tone of voice: """ + str(toneofvoice) + """.
        I want you to propose next 15 article titles with categories but be creative. Rotate categories randomly.
        Focus on articles with long-tail keywords, have high potential to rank in top positions on Google 
        and titles will have high click through rate. Read the categories from description in random order.
        The titles will have maximum of 9 words.
        If you want to use year, it should be 2023 but make sure it makes sense to use year!
        Output all in JSON format. Strictly follow the following JSON format:
        
        {
            "Category": "Category name",
            "Title": "Article title"
        }
        """
    else:
        prompt = """I have the following blog with the following description: """ + description + """.
                The blog has following articles with titles: """ + str(published_articles) + """.
                The blog has following tone of voice: """ + str(toneofvoice) + """.
                I want you to propose next 3 article titles with categories but be creative. Rotate categories randomly.
                Focus on articles with following keywords: """ +str(keywords)+""". You can create and use long-tail version of them. 
                Titles will have high click through rate. Read the categories from description in random order.
                The titles will have maximum of 9 words.
                If you want to use year, it should be 2023 but make sure it makes sense to use year!
                Output all in JSON format. Strictly follow the following JSON format:

                {
                    "Category": "Category name",
                    "Title": "Article title"
                }
                """

    chunked_output = ""
    for chunk in openai.ChatCompletion.create(
            model=cfg.llm_model,
            temperature=0.7,
            stream=True,
            messages=[
                {'role': 'system', 'content': 'Your name is BloggingGPT.'},
                {'role': 'system', 'content': "You're an expert in blogging, research and SEO."},
                {'role': 'system', 'content': 'You strictly return content user asked for only.'},
                {"role": "user", "content": prompt}
            ]
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            #print(content, end='')
            chunked_output += content

    return chunked_output