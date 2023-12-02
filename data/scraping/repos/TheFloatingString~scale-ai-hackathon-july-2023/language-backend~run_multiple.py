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

user_resp = input("Enter hash codes separated by a '>': ")

hash_ids = user_resp.split(">")

def get_hash(post_id):
    return mongo_client.db.coll.find_one({"_id": post_id})

for hash_id in hash_ids:
    print(get_hash(hash_id))