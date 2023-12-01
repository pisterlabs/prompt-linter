from gpt_index import Document, GPTSimpleVectorIndex
import streamlit as st
import openai
import os
openai.api_key = st.secrets["openai_api_key"]
openai.organization = st.secrets["openai_organization"]
assert openai.api_key is not None, "OpenAI API key not found"
os.environ["OPENAI_API_KEY"] = openai.api_key
os.environ["OPENAI_ORGANIZATION"] = openai.organization
documents = []
books = [
    "On_Love_And_Loneliness.txt",
    "The_Book_Of_Life.txt",
    "Total_Freedrom.txt"
]
for file in books:
    with open(file, "r") as f:
        documents.append(Document(f.read(), doc_id=file))

index = GPTSimpleVectorIndex(documents)
index_file_name = "index.json"
# save to disk
index.save_to_disk(index_file_name)