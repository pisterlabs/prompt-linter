from invoke import task
from quinn_gpt.scrapers import DocsScraper
from quinn_gpt.db import QuinnDB

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredHTMLLoader

from tqdm import tqdm

import os

VERSION = '5.1'
PERSIST_DIR = f'./chromadb/quinn_gpt'
qdb = QuinnDB('quinn_gpt')
scraper = DocsScraper(VERSION, qdb)


@task
def run(c, url):
    scraper.scrape_url(url, VERSION)

@task
def run_all(c):
    start_url = f'https://docs.unrealengine.com/{VERSION}/en-US/'
    scraper.crawl_site(start_url)

@task
def cache_to_chroma(c, chunk_size=400, reset=True):
    # load the document and split it into chunks
    chroma = Chroma(persist_directory=PERSIST_DIR, embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
    chroma.persist()

    for filename in tqdm(os.listdir('.cache')):
        loader = UnstructuredHTMLLoader(".cache/"+filename)
        documents = loader.load()

        # split it into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(chunk_size), chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # load it into Chroma
        chroma.add_documents(docs)

@task
def query(c, query, k=5):
    chroma = Chroma(persist_directory=PERSIST_DIR, embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
    results = chroma.similarity_search(query, k=k)
    for result in results:
        print(result.page_content)

@task
def estimate_cost(c):
    # Loops through all files in .cache and estimates the cost of embedding them

    total_cost = 0
    total_words = 0
    total_tokens = 0
    for filename in os.listdir('.cache'):
        with open(f'.cache/{filename}', 'r') as f:
            text = f.read()
            words = len(text.split())
            tokens = words*1.3
            total_tokens += tokens
            total_words += words
            cost = tokens / 1000 *.0001
            total_cost += cost
    
    print(f'{total_words} words, ${total_cost}')


@task
def test(c):
    c.run('pytest ./tests --cov=quinn_gpt --cov-report=term-missing')

@task
def remove_pound(c):
    qdb.remove_hashed_urls()