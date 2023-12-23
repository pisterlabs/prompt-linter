import os
import torch
import torch.nn
from openai import OpenAI
import inspect
import tiktoken
import chromadb
from loguru import logger
logger.add("get_embeddings.log", rotation="500 MB")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

def calculate_embedding(text, model="text-embedding-ada-002"):
    client = OpenAI(api_key=os.getenv('OPENAI_KEY'))
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_source_code(api_name):
    try:
        # Import the module and get the attribute (function/class)
        api = eval(api_name) # dangerous

        # Get the source code
        source_code = inspect.getsource(api)
        return source_code
    except (ImportError, AttributeError):
        logger.error(f"API '{api_name}' not found.")
        return None
    except TypeError:
        logger.error(f"Source code for '{api_name}' is not available (might be a built-in or C-extension).")
        return None

api_list = []
def get_api_list(pkg, pkg_name):
    for api in dir(pkg):
        api_list.append(f"{pkg_name}.{api}")

chroma_client = chromadb.PersistentClient(path="../embedding-db")
collection = chroma_client.get_or_create_collection(name="torch.nn")
# get_api_list(torch)
get_api_list(torch.nn, pkg_name='torch.nn')

for api in api_list:
    ids = collection.get(ids=[api])['ids']
    if ids == []:
        
        source_code = get_source_code(api)
        if source_code == None:
            continue
        source_code_token_length = num_tokens_from_string(source_code, 'cl100k_base')
        if source_code_token_length > 8192:
            logger.warning(f"Too long. There are {source_code_token_length} tokens in {api}.")
            continue
        logger.info(f"There are {source_code_token_length} tokens in {api}.")
        
        embedding = calculate_embedding(source_code, model='text-embedding-ada-002')
        
        collection.add(
            embeddings=[embedding,],
            documents=[source_code,],
            metadatas=[{"source": api},],
            ids=[api,]
        )