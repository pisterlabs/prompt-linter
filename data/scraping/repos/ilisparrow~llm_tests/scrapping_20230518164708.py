from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import json

#Setup env vars : 
load_dotenv()

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

#TODO : DO URL Check and show message when not valid
#Web Scrapping and 
url_to_watch = st.text_input("Input your url here","https://laion.ai/blog/")#UI

html = urllib.request.urlopen(url_to_watch).read()
text_from_webpage = text_from_html(html)

#Logging
file_path = "output.txt"
with open(file_path, "w") as file:
    file.write(text_from_webpage)
print("Variable content saved to the file:", file_path)

#LLM part
#if st.button('Analyze'):
prompt = PromptTemplate(
    input_variables=["webpage"],
    template="In this web page, can you find a pattern, list all the articles and their publication dates. Limit yourself to the first 2. In Json format. No Other text.\
        webpage :  \"{webpage}\"",
    )
llm = OpenAI(temperature=0.0)
prompt_to_send = prompt.format(webpage=text_from_webpage)
result_from_chatgpt = llm(prompt_to_send).replace("\n", "")

file_path = "gpt_out.txt"
with open(file_path, "w") as file:
    file.write(result_from_chatgpt)
print("Variable content saved to the file:", file_path)


parsed_data = json.loads(result_from_chatgpt)

st.write(result_from_chatgpt)

