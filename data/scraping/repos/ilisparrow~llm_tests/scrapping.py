from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import json
from dotenv import dotenv_values
from googlesearch import search
#Setup env vars :
load_dotenv()
HARD_LIMIT_CHAR = 10000 
env_vars = dotenv_values(".env")
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)

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

#TODO : DO URL Check and show message when not valid
#Web Scrapping and 
url_to_watch = st.text_input("Input your url here","https://laion.ai/blog/")#UI

html = urllib.request.urlopen(url_to_watch).read()
text_from_webpage = text_from_html(html)
#TODO : Fixe this limit, in a smarter way
text_from_webpage = text_from_webpage[:HARD_LIMIT_CHAR]

#Logging
file_path = "output.txt"
with open(file_path, "w") as file:
    file.write(text_from_webpage)
print("Variable content saved to the file:", file_path)

#LLM part
#if st.button('Analyze'):
prompt = PromptTemplate(
    input_variables=["webpage"],
    template="In this web page, can you find a pattern, list all the articles and their publication dates. Do not mix the date with the reading time. Limit yourself to the first 3. In Json format, using these keys \"title\", \"date\". No Other text. \
        webpage :  \"{webpage}\"",
    )
llm = OpenAI(openai_api_key=env_vars['OPENAI_API_KEY'],temperature=0.9)
prompt_to_send = prompt.format(webpage=text_from_webpage)
result_from_chatgpt = llm(prompt_to_send).replace("\n", "")
print(result_from_chatgpt)
file_path = "gpt_out.txt"
 

parsed_articles = json.loads(result_from_chatgpt)
#Logging
file_path = "output_gpt.txt"
with open(file_path, "w") as file:
    file.write(result_from_chatgpt)
print("Variable content saved to the file:", file_path)

#st.json(parsed_articles)
text_to_watch_for = st.text_input("What should we look for ?","ex. investments or Taiwan")#UI

for article in parsed_articles: 
    print("--------------------------")
    print(article["title"])
    query = article["title"]


    for j in search(query, tld="co.in", num=1, stop=1, pause=2):
        print(j)

    st.header(article["title"])
    st.text(article["date"])

#TODO : Do  a google search limited to the websited given, of the articles, get their content
#TODO : Add a field to ask a quetion (maybe multiple choice field)
#TODO : Ask the article and the question to Chatgpt 
#TODO : Display results to the user
#TODO : 
