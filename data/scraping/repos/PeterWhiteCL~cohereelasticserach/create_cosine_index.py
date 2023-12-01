import json
import hashlib
from elasticsearch import Elasticsearch
import numpy as np

# Add the necessary imports
from config import COHERE_MODEL, DATA_PATH, EMBED_COLUMN, COHERE_API_KEY
from utils import get_cohere_embedding

# Replace with your Elasticsearch URL
es_url = "http://localhost:9200"

# Elasticsearch index name and field name for the embeddings
index_name = "abstract_index"  # Replace with your index name
embedding_field = "embedding_vector"  # Replace with the name of the field containing the embeddings

# Default k value for k-NN search
default_k = 3

# Create the Elasticsearch client
es = Elasticsearch([es_url],basic_auth=('elastic', COHERE_API_KEY))

# Create the Elasticsearch index (if it doesn't exist) with the correct mapping
if not es.indices.exists(index=index_name):
    index_mapping = {
  "mappings": {
    "properties": {
      "abstract": {
        "type": "text"
      },
      "embedding-vector": {
        "type": "dense_vector",
        "dims": 1024,
        "index": "true",
        "similarity": "cosine"
      }
    }
  }
}
    es.indices.create(index=index_name, body=index_mapping)

# Populate the index from the cache.jsonl file
with open("cache.jsonl", "r") as file:
    data = json.load(file)
    for abstract_text, embedding_vector in data.items():
        # Index the document in Elasticsearch
        doc_id = hashlib.sha256(abstract_text.encode()).hexdigest()
        embedding_vector = [float(x) for x in embedding_vector]
        es.index(index=index_name, id=doc_id, document={embedding_field: embedding_vector})

# Use the provided test_abstract to get the query_vector
test_abstract = "In this work, we explore \"prompt tuning\", a simple yet effective mechanism for learning \"soft prompts\" to condition frozen language models to perform specific downstream tasks. Unlike the discrete text prompts used by GPT-3, soft prompts are learned through backpropagation and can be tuned to incorporate signal from any number of labeled examples. Our end-to-end learned approach outperforms GPT-3's \"few-shot\" learning by a large margin. More remarkably, through ablations on model size using T5, we show that prompt tuning becomes more competitive with scale: as models exceed billions of parameters, our method \”closes the gap\” and matches the strong performance of model tuning (where all model weights are tuned). This finding is especially relevant in that large models are costly to share and serve, and the ability to reuse one frozen model for multiple downstream tasks can ease this burden. Our method can be seen as a simplification of the recently proposed \”prefix tuning\” of Li and Liang (2021), and we provide a comparison to this and other similar approaches. Finally, we show that conditioning a frozen model with soft prompts confers benefits in robustness to domain transfer, as compared to full model tuning."
test = get_cohere_embedding(test_abstract, model_name=COHERE_MODEL, input_type="search_document")
query_vector = [float(x) for x in embedding_vector]

# Perform k-NN search based on cosine similarity with the provided query_vector
query = {
  "knn": {
    "field": "embeddding-vector",
    "query_vector": query_vector,
    "k": 3,
    "num_candidates": 100
  }
}
result = es.knn_search(index=index_name, body=query)

# Process the search results
for hit in result["hits"]["hits"]:
    score = hit["_score"]
    embedding_vector = hit["_source"][embedding_field]
    print(f"Score: {score:.4f}, Embedding: {embedding_vector}")




# Default k value for k-NN search
default_k = 1

def search(query):

    # Create the Elasticsearch client
    es = Elasticsearch([es_url], basic_auth=(<elastic_user>, <elastic_password>))

    # Use the provided query to get the query_vector
    test_abstract = query
    test_embedding = get_cohere_embedding(test_abstract, model_name=COHERE_MODEL, input_type="search_query")
    query_vector = [float(x) for x in test_embedding]

    # Perform k-NN search based on cosine similarity with the provided query_vector
    query = {
         "knn": {
            "field": "embedding-vector",
            "query_vector": query_vector,
            "k": 3,
            "num_candidates": 100
        },
 
    }
    result = es.search(index=index_name, body=query)

    matches = [{"abstract": data_dict[str(hit["_source"][embedding_field])], "score": hit["_score"]}
               for hit in result.body["hits"]['hits'][:3]]
    return matches

