# https://www.paepper.com/blog/posts/build-q-and-a-bot-of-your-website-using-langchain/
import json
from loguru import logger
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import os
import chromadb
import random
import string


# chroma Client
client = chromadb.PersistentClient(path="./db")
collection = client.create_collection("health")

# Env Area
os.environ["OPENAI_API_KEY"] = "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"

# json loader function


def load_json(file):
    logger.info(f'Loading JSON file: {file}')
    with open(file) as f:
        data = json.load(f)
    return data

def generate_random_string(length):
    # Combine all the letters (both lowercase and uppercase) and digits
    possible_characters = string.ascii_letters + string.digits

    # Use random.choices() to pick characters, then use ''.join() to combine them into a single string
    random_string = ''.join(random.choice(possible_characters) for i in range(length))
    
    return random_string

def main():
    # Load JSON file
    pages = load_json('pages.json')
    # logger.info(f'JSON file loaded: {data}')

    # Create Chunk For Database
    text_splitter = CharacterTextSplitter(chunk_size=1000, separator="\n")
    docs, metadatas, ids = [], [], []
    #i = 1
    for page in pages:
        splits = text_splitter.split_text(page['text'])
        docs.extend(splits)
        metadatas.extend([{"source": page['source']}] * len(splits))
        #ids.append(generate_random_string(10))
        logger.debug(f"Split {page['source']} into {len(splits)} chunks")
        #i += 1

    # Create Embedding
    # store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
    # with open("faiss_store.pkl", "wb") as f:
    #     pickle.dump(store, f)

    for meta in metadatas:
        print(meta)
        ids.append(generate_random_string(10))

    # Chroma
    collection.add(
        documents=docs,
        metadatas=metadatas,
        ids=ids
    )


# if main
if __name__ == '__main__':
    main()
