import markdown2
from bs4 import BeautifulSoup
from transformers import GPT2TokenizerFast
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import os

import numpy as np
from nltk.tokenize import sent_tokenize

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Open the markdown file
with open("tapis/actors.md", "r") as file:
    content = file.read()

# Use markdown2 to convert the markdown file to html
html = markdown2.markdown(content)

# Use BeautifulSoup to parse the html
soup = BeautifulSoup(html, "html.parser")

# Initialize variables to store heading, subheading, and corresponding paragraphs
headings = []
paragraphs = []

data = []

MAX_WORDS = 500

def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))

docs = []
metadatas = []
# Iterate through the tags in the soup
for tag in soup.descendants:
    # Check if the tag is a heading
    if tag.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        # When the next heading is encountered, print the heading, subheading, and corresponding paragraphs
        if headings and paragraphs:
            hdgs = " ".join(headings)
            para = " ".join(paragraphs)
            docs.append(para)
            metadatas.extend([{"source": hdgs}])
            headings = []
            paragraphs = []
        # Add to heading
        headings.append(tag.text)
    # Check if the tag is a paragraph
    elif tag.name == "p":
        paragraphs.append(tag.text)



# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
