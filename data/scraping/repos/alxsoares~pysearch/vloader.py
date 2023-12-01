import pandas as pd
import tiktoken
import faiss
import re
import numpy as np
from ai import AI

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from faiss import write_index, read_index

ai = AI()
#embedding_dim = 512
#findex = faiss.IndexFlatL2(embedding_dim)

df = pd.read_csv('data.csv')

def get_token_count(string: str) -> int:
    encoding_name = "cl100k_base"
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


houses = ""
for index, row in df.iterrows():
    features = row['features'].replace("[", "").replace("]","")
    features = re.split(r",(?=')", features)
    houses += "Home available for sale at " + row['address'] + " " + row['city_state'] + " costs " + row['price'] + ", " + row['description'] + " has " + " ".join(features)+"\n"
 
print(houses)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_text(houses)
print(docs)
embeddings = ai.get_open_ai_embedding()
#db = FAISS.from_texts(houses, embedding=embeddings)
db = FAISS.from_texts(docs, embedding=embeddings)
print(db)
db.save_local('houses_new.index')