import os
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup

from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

import pandas as pd
import numpy as np
from numpy.linalg import norm

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_raw_content(url):
  response = requests.get(url)

  soup = BeautifulSoup(response.content, 'html.parser')

  content_div = soup.find('div',{ 'class': 'mw-parser-output'})

  unwanted_tags = ['sup','span','table','ul','ol']

  for tag in unwanted_tags:
    for match in content_div.findAll(tag):
      match.extract()

  return content_div.get_text()

def get_text_chunks(content):
  text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show
    chunk_size = 100,
    chunk_overlap = 20,
    length_function = len,
  )

  return text_splitter.create_documents([content])

def get_embeddings(text,  model="text-embedding-ada-002"):
  text = text.replace("\n"," ")
  return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

print("1. Get content")

raw_content = get_raw_content("https://es.wikipedia.org/wiki/Operaci%C3%B3n_Chav%C3%ADn_de_Hu%C3%A1ntar")

print("2. Get chunks")
chunks = get_text_chunks(raw_content)


print("3. Get embeddings")
text_chunks = []

for text in chunks:
  text_chunks.append(text.page_content)

df = pd.DataFrame({'text_chunks': text_chunks})
df["ada_embedding"] = df.text_chunks.apply(lambda x: get_embeddings(x,
                                                                    model="text-embedding-ada-002"))

users_question = "Fecha de la operacion"
question_embedding = get_embeddings(text=users_question, model="text-embedding-ada-002")

cos_sim = []

print("4. Calculate cosine")
for index, row in df.iterrows():
  A = row.ada_embedding
  B = question_embedding

  cosine = np.dot(A,B)/(norm(A)*norm(B))
  cos_sim.append(cosine)

df["cos_sim"] = cos_sim
df.sort_values(by=["cos_sim"], ascending=False)

print(df)

