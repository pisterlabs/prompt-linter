from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import OpenAIEmbeddings

documents = [
    "The cat is on the mat.",
    "There is a cat on the mat.",
    "The dog is in the yard.",
    "There is a dog in the yard.",
]

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

document_embeddings = embeddings.embed_documents(documents)

query = "The cat is in the yard."
query_embedding = embeddings.embed_query(query)

similarity_scores = cosine_similarity([query_embedding], document_embeddings)[0]

most_similar_index = np.argmax(similarity_scores)
most_similar_document = documents[most_similar_index]

print(f"Most similar document to the query {query}: {most_similar_document}")

from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

documents = ["Document 1", "Document 2", "Document 3"]
document_embeddings = hf.embed_documents(documents)