from flask import Flask, request
import requests
import os
import openai
from dotenv import load_dotenv
from pymongo import MongoClient

from flask_cors import CORS

from flask_cors import CORS, cross_origin

from scipy.spatial import distance

import chromadb
from chromadb.utils import embedding_functions
'''
chroma_client = chromadb.Client()

default_ef = embedding_functions.DefaultEmbeddingFunction()




collection = chroma_client.create_collection(name="my_collection")

collection.add(
    documents=["This is a document", "This is another document"],
    metadatas=[{"source": "my_source"}, {"source": "my_source"}],
    ids=["id1", "id2"]
)
'''

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY_SAI")

mongo_client = MongoClient("mongodb+srv://user:pass@cluster0.gtjf1ql.mongodb.net/?retryWrites=true&w=majority")

truth_es = input("Type a sentence in another language: ")

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"Translate the following text to English, with nothing else returned other than the pure translation: {truth_es}"}
  ]
)

truth_en = completion.choices[0]["message"]["content"]

emb_truth_en = openai.Embedding.create(
  model="text-embedding-ada-002",
  input=truth_en
)

## print(completion.choices[0].message["content"])

# emb_truth_es["data"][0]["embedding"]

def scale_score(x):
    return (0.3-x)/(0.3)

for i in range(10):
    user_pred = input("User prediction: ")

    if user_pred=="q":
        break

    emb_user_pred = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=user_pred
            )

    cos_dist = round(scale_score(distance.cosine(emb_truth_en["data"][0]["embedding"], 
                               emb_user_pred["data"][0]["embedding"])
    ),3)

    print(cos_dist)

print(truth_en)
