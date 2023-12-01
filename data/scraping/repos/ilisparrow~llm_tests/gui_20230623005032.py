## IMPORTS

from bs4 import BeautifulSoup              # Importing BeautifulSoup for HTML parsing
from bs4.element import Comment             # Importing Comment class for extracting comments from HTML
import urllib.request                       # Importing urllib.request for making HTTP requests
import streamlit as st                      # Importing streamlit for building interactive web apps
import os                                   # Importing os for accessing operating system functionalities
from dotenv import load_dotenv              # Importing load_dotenv for loading environment variables
from langchain.llms import OpenAI            # Importing OpenAI class from langchain.llms module
from langchain.prompts import PromptTemplate # Importing PromptTemplate class from langchain.prompts module
import json                                 # Importing json module for working with JSON data
from dotenv import dotenv_values            # Importing dotenv_values for loading environment variables from .env file
from googlesearch import search             # Importing search function from googlesearch module
import requests                            # Importing requests module for making HTTP requests


## SETUP ENVIRONMENT VARIABLES

load_dotenv()
env_vars = dotenv_values(".env") 


## Define system relevant input data for application
HARD_LIMIT_CHAR = 10000

## Functions

def tag_visible(element):
    excluded_tags = ['a', 'style', 'script', 'head', 'title', 'meta', '[document]']

    if element.parent.name in excluded_tags:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.find_all(text=tag_visible)
    visible_texts = [t.strip() for t in texts if t.strip()]

    return " ".join(visible_texts)


def extract_json_values(input_str):
    results = []
    while input_str:
        try:
            value = json.loads(input_str)
            input_str = ""
        except json.decoder.JSONDecodeError as exc:
            if str(exc).startswith("Expecting value"):   
                input_str = input_str[exc.pos+1:]
                continue
            elif str(exc).startswith("Extra data"):
                value = json.loads(input_str[:exc.pos])
                input_str = input_str[exc.pos:]
        results.append(value)
    return results

## Process website and save content to file
def process_website(url, output_file_name):
    html = urllib.request.urlopen(url).read()
    text_from_webpage = text_from_html(html)
    text_from_webpage = text_from_webpage[:HARD_LIMIT_CHAR]

    # Logging
    file_path = output_file_name
    with open(file_path, "w") as file:
        file.write(text_from_webpage)
    print("Variable content saved to the file:", file_path)
    return text_from_webpage

def get_link_based_on_article_name_via_google(article_title):
    search = article_title
    url = 'https://www.google.com/search'

    headers = {
        'Accept' : '*/*',
        'Accept-Language': 'en-US,en;q=0.5',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.82',
    }
    parameters = {'q': search}

    content = requests.get(url, headers = headers, params = parameters).text
    soup = BeautifulSoup(content, 'html.parser')

    search = soup.find(id = 'search')
    first_link = search.find('a')
    article_link= first_link['href']
    return first_link['href']


def prompt_to_llm_response(text_from_webpage, prompt_input):
    prompt = PromptTemplate(
        input_variables=["webpage", "prompt_text"],
        template="\"{prompt_text}\" \
            webpage :  \"{webpage}\"",
    )
    prompt_to_send = prompt.format(webpage=text_from_webpage, prompt_text=prompt_input)

    llm = OpenAI(openai_api_key=env_vars['OPENAI_API_KEY'], temperature=0.9)
    result_from_chatgpt = llm(prompt_to_send).replace("\n", "").replace("Answer:","")

    return result_from_chatgpt



url_to_watch = st.text_input("Input your url here","https://laion.ai/blog/")
## Process website and save content to file
text_from_webpage = process_website(url_to_watch, "output.txt")
text_from_webpage = text_from_webpage[:HARD_LIMIT_CHAR]


prompt_news = "In this web page, can you find a pattern, list all the article titles and their publication dates. Do not mix the date with the reading time. Limit yourself to the first 3. In Json format, using these keys \"title\", \"date\". No Other text."
result_from_chatgpt = prompt_to_llm_response(text_from_webpage,prompt_news)

st.json(json.dumps(json.loads(result_from_chatgpt), indent=4))

print(json.dumps(json.loads(result_from_chatgpt), indent=4))
#print(result_from_chatgpt)
