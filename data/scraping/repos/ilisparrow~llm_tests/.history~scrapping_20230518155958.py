from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI

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
url_to_watch = st.text_input("Input your url here","https://laion.ai/blog/")


html = urllib.request.urlopen(url_to_watch).read()
text = text_from_html(html)


file_path = "output.txt"

# Open the file in write mode and write the text content
with open(file_path, "w") as file:
    file.write(text)

print("Variable content saved to the file:", file_path)
llm = OpenAI(temperature=0.9)
text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))


st.write('The current movie title is')

