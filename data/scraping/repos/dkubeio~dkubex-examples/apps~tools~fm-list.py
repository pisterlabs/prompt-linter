# script.py
# import ray
import glob
import os
import weaviate  # weaviate-python client
import argparse
from typing import Tuple

import sys
#sys.path.append('/usr/local/lib/python3.10/dist-packages')


from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import Weaviate

'''
Weaviate URL from inside the cluster
--- "http://weaviate.d3x.svc.cluster.local/"
Weaviate URL from outside the cluster
--- NodePort - "http://<ip>:30716/api/vectordb/"
--- Loadbalancer - "http://<ip>:80/api/vectordb/"
'''

WEAVIATE_URL = os.getenv("WEAVIATE_URI", None)

def get_vectordb_client() -> VectorStore:
    DKUBEX_API_KEY = os.getenv( "DKUBEX_API_KEY", "deadbeef")

    # Use Weaviate VectorDB
    weaviate_client = weaviate.Client(
        url=WEAVIATE_URL,
        additional_headers={"Authorization": DKUBEX_API_KEY},
    )

    return weaviate_client

if __name__ == "__main__":

    client = get_vectordb_client()
    schema = client.schema.get()

    all_classes = [c["class"] for c in schema["classes"]]
    #print(f"Classes: {all_classes}")

    for c in all_classes:
        r = client.query.aggregate(class_name=c).with_meta_count().do()
        #print(r)

        if "docs" in c:
            dataset = c[1:-4]
            ctype = "Documents"
        elif "chunks" in c:
            dataset = c[1:-4]
            ctype = "Chunks"
        else:
            dataset = c
            ctype = "unkown"

        n_docs = r["data"]["Aggregate"][c][0]["meta"]["count"]

        
        print(f"Dataset: {dataset} -- Number of {ctype}: {n_docs}")

