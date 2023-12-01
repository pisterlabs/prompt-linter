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


def get_embeddings() -> Embeddings:
    # Use HuggingFace model for embeddings
    '''
    from langchain.embeddings import HuggingFaceEmbeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}

    hf = HuggingFaceEmbeddings(
        model_name=model_name,
    )
    '''
    from langchain.embeddings import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(client=None)
    return embeddings

def get_vectordb_client() -> VectorStore:
    DKUBEX_API_KEY = os.getenv( "DKUBEX_API_KEY", "deadbeef")

    # Use Weaviate VectorDB
    weaviate_client = weaviate.Client(
        url=WEAVIATE_URL,
        additional_headers={"Authorization": DKUBEX_API_KEY},
    )

    return weaviate_client

def create_paperqa_vector_indexes(client:VectorStore, embeddings:Embeddings, dataset:str) -> Tuple[VectorStore, VectorStore]:

    # Create 2 classes (or DBs) in Weaviate
    # One for docs and one for chunks
    docs = Weaviate(
        client=client,
        index_name="D"+ dataset + "docs",
        text_key="paperdoc",
        attributes=['dockey'],
        embedding=embeddings,
    )


    chunks = Weaviate(
        client=client,
        index_name="D"+ dataset + "chunks",
        text_key="paperchunks",
        embedding=embeddings,
        attributes=['doc', 'name'],
    )

    return docs, chunks


'''
@ray.remote
'''
def query(dataset):    

    from paperqa import Docs

    embeddings = get_embeddings()
    weaviate = get_vectordb_client()
    docs, chunks = create_paperqa_vector_indexes(weaviate, embeddings, dataset)
    docs_store = Docs(doc_index=docs, texts_index = chunks)
    docs_store.build_doc_index()

    while True:
        query = input("Question>: ")
        if query.lower() in ['exit','stop','quit']:
            exit(1)
        answer = docs_store.query(query)
        print(answer)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", "-d", required=True, type=str, help="A name to represent the ingested docs",
    )

    args = parser.parse_args()
    import re
 
    dataset_pat = r"[A-Za-z0-9]+"
 
    if re.fullmatch(dataset_pat, args.dataset) is None:
        print(f"{args.dataset} should be alphanumeric")
        exit(1)



    '''
    # Automatically connect to the running Ray cluster.
    ray.init()
    print(ray.get(ingest(source, dataset).remote()))
    '''

    query(args.dataset)
