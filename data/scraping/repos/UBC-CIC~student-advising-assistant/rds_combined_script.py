from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import FakeEmbeddings
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import json
import pickle
import os
import shutil
import pathlib
import ast
import doc_loader
import batcher
import torch
from combined_embeddings import concat_embeddings
import sys
import argparse
sys.path.append('..')
from aws_helpers.param_manager import get_param_manager
from aws_helpers.s3_tools import download_s3_directory, upload_directory_to_s3
from aws_helpers.ssh_forwarder import start_ssh_forwarder

### CONFIGURATION
# AWS Secrets Manager config for the Pinecone secret API key and region
secret_name = "credentials/RDSCredentials"
param_manager = get_param_manager()

# Config for the index
index_config = {
    "name": "documents_index", # name of the table in the RDS DB
    "namespace": "all-mpnet-base-v2",
    "base_embedding_model": "sentence-transformers/all-mpnet-base-v2", # can be changed to any huggingface model name
    "base_embedding_dimension": 768, # depends on the base embedding model
    "embeddings": [
        "parent_title_embeddings",
        "title_embeddings",
        "document_embeddings"
    ]
}

### ARG CONFIG
parser = argparse.ArgumentParser()
parser.add_argument('--compute_embeddings', dest='compute_embeddings', action='store_true')
parser.add_argument('--no-compute_embeddings', dest='compute_embeddings', action='store_false')
parser.set_defaults(compute_embeddings=True)
parser.add_argument('--clear_index', dest='clear_index', action='store_true')
parser.add_argument('--no-clear_index', dest='clear_index', action='store_false')
parser.set_defaults(clear_index=True)
parser.add_argument('--gpu_available', dest='gpu_available', action='store_true')
parser.add_argument('--no-gpu_available', dest='gpu_available', action='store_false')
parser.set_defaults(gpu_available=False)

args = parser.parse_args()

print(args)

if not args.gpu_available:
    torch.set_num_threads(os.cpu_count())
    
### DOCUMENT LOADING 

# Load the csv of documents from s3
docs_dir = 'documents' 
download_s3_directory(docs_dir)
docs = doc_loader.load_docs(os.path.join(docs_dir, "website_extracts.csv"), eval_strings=False)
metadatas = [doc.metadata for doc in docs]
ids = [doc.metadata['doc_id'] for doc in docs]

# Load precomputed embeddings
embed_dir = f"embeddings-{index_config['namespace']}"
if not args.compute_embeddings: 
    download_s3_directory(embed_dir)

# Create the different lists of texts for embedding
title_sep = ' : ' # separator to use to join titles into strings
parent_titles = [title_sep.join(doc.metadata['parent_titles']) for doc in docs]
titles = [title_sep.join(doc.metadata['titles']) for doc in docs]
texts = [doc.page_content for doc in docs]

### CREATE EMBEDDINGS (DENSE VECTORS)

# Lists of embeddings
embedding_names = ['parent_title_embeddings', 'title_embeddings', 'document_embeddings']
embedding_texts = [parent_titles,titles,texts]

# For each embedding, compute the embedding and save to pickle file
embeddings = {}
if not args.compute_embeddings:
    for file in pathlib.Path(embed_dir).glob('*.pkl'):
        with open(file, "rb") as f:
            data = pickle.load(f)
            embeddings[file.stem] = data['embeddings']
            print(f'Loaded embeddings {file.stem}')
else:
    # Load the base embedding model from huggingface
    if args.gpu_available:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    else:
        device = "cpu"
    print("Torch device is: ", device)    
    base_embeddings = HuggingFaceEmbeddings(model_name=index_config['base_embedding_model'], model_kwargs={'device': device})
    base_embeddings = HuggingFaceEmbeddings(model_name=index_config['base_embedding_model'], 
                                        model_kwargs={'device': device},
                                       encode_kwargs={
                                            'show_progress_bar': True,
                                            'batch_size': 64})

    os.makedirs(embed_dir,exist_ok=True)
    
    ### Create dense vectors
    for name,content in zip(embedding_names,embedding_texts):
        print(f'Computing {name}')
        embeddings[name] = base_embeddings.embed_documents(content)

        print(f'Saving {name} to directory')
        with open(os.path.join(embed_dir, f'{name}.pkl'), "wb") as f:
            pickle.dump({'embeddings': embeddings[name]}, f, protocol=pickle.HIGHEST_PROTOCOL)
    

### CREATE PGVECTOR INDEX

# Save index config to json
index_dir = 'indexes'
pgvector_dir = os.path.join(index_dir,'pgvector')
os.makedirs(pgvector_dir,exist_ok=True)
with open(os.path.join(pgvector_dir,'index_config.json'),'w') as f:
    json.dump(index_config,f)

# Connect to the rds db
db_secret = param_manager.get_secret(secret_name)

forwarder_port = None
if "MODE" in os.environ and os.environ["MODE"] == "dev":
    # Use an SSH forwarder so that we can connect to the pgvector RDS in a private subnet
    try:
        server = start_ssh_forwarder(db_secret["host"],db_secret["port"])
        forwarder_port = server.local_bind_port
    except Exception as e:
        print(f'Could not set up ssh forwarder for local connection to rds: {str(e)}')
    
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    database=db_secret["dbname"],
    user=db_secret["username"],
    password=db_secret["password"],
    host='localhost' if forwarder_port else db_secret["host"],
    port=forwarder_port if forwarder_port else db_secret["port"],
)

# Upload the embedded documents
embedding_list = [embeddings[name] for name in index_config['embeddings']]
combined_embeddings = concat_embeddings(embedding_list)
embeddings = {} # Don't need to keep embeddings in memory
fake_embeddings_model = FakeEmbeddings(size=index_config['base_embedding_dimension']*len(index_config['embeddings']))
# ^ Used to create pgvector db, don't need real embeddings model since precomputed

print('Begin upload to db with pgvector')
db = PGVector.from_embeddings(
    text_embeddings=list(zip(texts,combined_embeddings)),
    embedding=fake_embeddings_model,
    metadatas=metadatas,
    ids=ids,
    collection_name=index_config['name'],
    connection_string=CONNECTION_STRING,
    pre_delete_collection=args.clear_index
)
print('Finished upload to db with pgvector')

### UPLOAD TO S3 & CLEANUP
    
# Upload documents to s3
upload_directory_to_s3(embed_dir)
upload_directory_to_s3(docs_dir)
upload_directory_to_s3(index_dir)

# Delete directories from disk
shutil.rmtree(docs_dir)
shutil.rmtree(embed_dir)
shutil.rmtree(index_dir)