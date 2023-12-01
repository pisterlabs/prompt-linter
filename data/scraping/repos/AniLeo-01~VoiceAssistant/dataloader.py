import os
import requests
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import re
from dotenv import load_dotenv
from scraper import get_documentation_urls, scrape_all_content

load_dotenv()
dataset_path = os.environ.get('DEEPLAKE_DATASET_PATH')
activeloop_token = os.environ.get('ACTIVELOOP_TOKEN')

embeddings = OpenAIEmbeddings()

def load_docs(root_dir, filename):
    docs = []
    try:
        loader = TextLoader(os.path.join(
            root_dir, filename), encoding='utf-8'
            )
        docs.extend(loader.load_and_split())
    except Exception as e:
        pass
    return docs

def split_docs(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)

def create_deeplake(embedding_function, texts):
    db = DeepLake(dataset_path=dataset_path, embedding_function=embedding_function)
    db.add_documents(texts)

def load_deeplake(embedding_function):
    db = DeepLake(dataset_path=dataset_path, embedding_function=embedding_function, read_only=True)
    return db

def create_dataset_and_load_datalake():
    #initialize scrapers and dataloaders
    base_url = 'https://huggingface.co'
    filename='content.txt' 
    root_dir ='./'
    relative_urls = get_documentation_urls()
    content = scrape_all_content(base_url, relative_urls,filename)
    docs = load_docs(root_dir,filename)
    texts = split_docs(docs)
    create_deeplake(embedding_function=OpenAIEmbeddings(), texts=texts)
    # Clean up by deleting the content file
    if os.path.exists(filename):
        os.remove(filename)

if __name__ == '__main__':
    create_dataset_and_load_datalake()