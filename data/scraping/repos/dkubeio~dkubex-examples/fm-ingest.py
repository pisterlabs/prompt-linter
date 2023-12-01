# script.py
# import ray
import glob
import os
import argparse
from typing import Tuple

import sys
#sys.path.append('/usr/local/lib/python3.10/dist-packages')


import weaviate  # weaviate-python client
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
def ingest(source, dataset):    

    from paperqa import Docs

    embeddings = get_embeddings()
    weaviate = get_vectordb_client()

    # Does this dataset already exist?
    schema = weaviate.schema.get()
    all_classes = [c["class"] for c in schema["classes"]]

    docs, chunks = create_paperqa_vector_indexes(weaviate, embeddings, dataset)
    docs_store = Docs(doc_index=docs, texts_index = chunks)
    if "D"+dataset+"docs" in all_classes:
        #raise Exception(f"The specified dataset {dataset}  already exists")
        docs_store.build_doc_index()

    docs_list = glob.glob(os.path.join(source, "**/*.pdf"), recursive=True)
    count = total = 0

    for doc in docs_list:
        #print("Ingesting doc: ", doc, end="  ")
        print("Ingesting doc: ", doc)
        total += 1
        try:
            docs_store.add(doc)
            print("----- Ingestion succeded")
            count += 1
        except Exception as ingest_except:
            print("xxxxx Ingestion failed, error: ", ingest_except)

    print(f"\n ------ Ingest {count} of {total} documents ------")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", "-s", required=True, type=str, help="The path where docs are sourced from for ingestion",
    )
    parser.add_argument(
        "--dataset", "-d", required=True, type=str, help="A name to represent the ingested docs",
    )

    args = parser.parse_args()
    import re
 
    source_pat = r"[A-Za-z0-9/-_\.]+"
    dataset_pat = r"[A-Za-z0-9]+"
 
    print(f"\nsource path: {args.source} ")
    print(f"dataset: {args.dataset} \n")
    # validate the inputs
    if re.fullmatch(dataset_pat, args.dataset) is None:
        print(f"{args.dataset} should be alphanumeric")
        exit(1)

    #if re.fullmatch(source_pat, args.source) is None:
    if os.path.exists(args.source) is False :
        print(f"{args.source} doesn't exist")
        exit(1)


    '''
    # Automatically connect to the running Ray cluster.
    ray.init()
    print(ray.get(ingest(source, dataset).remote()))
    '''

    # Workaround for openai-python bug #140. Doesn't close connections
    import warnings
    warnings.simplefilter("ignore", ResourceWarning)

    from datetime import datetime
    start_time = datetime.now()

    try:
        ingest(args.source, args.dataset)
    except Exception as ingest_failure:
        print(ingest_failure)
    
    print(' ------ Duration(hh:mm:ss) {} ------'.format(datetime.now() - start_time))
