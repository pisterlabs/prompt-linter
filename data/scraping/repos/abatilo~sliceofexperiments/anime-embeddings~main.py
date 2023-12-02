import os

import cohere
import torch
from datasets import load_dataset

co = cohere.Client(
    os.environ["COHERE_API_KEY"]
)  # Add your cohere API key from www.cohere.com

docs_stream = load_dataset(
    f"abatilo/myanimelist-embeddings", split="train", streaming=True
)

docs = []
doc_embeddings = []

for doc in docs_stream:
    docs.append(doc)
    doc_embeddings.append(doc["embedding"])

doc_embeddings = torch.tensor(doc_embeddings)

while True:
    query = input("What do you want to see?: ")

    response = co.embed(texts=[query], model="embed-multilingual-v2.0")
    query_embedding = response.embeddings
    query_embedding = torch.tensor(query_embedding)

    # Compute dot score between query embedding and document embeddings
    dot_scores = torch.mm(query_embedding, doc_embeddings.transpose(0, 1))
    top_k = torch.topk(dot_scores, k=3)

    for doc_id in top_k.indices[0].tolist():
        print(docs[doc_id]["title"])
        print(docs[doc_id]["synopsis"], "\n")
