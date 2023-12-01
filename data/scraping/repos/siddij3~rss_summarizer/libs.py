from bs4 import BeautifulSoup
import json
from selenium import webdriver
import re
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import requests
import openai

import tiktoken

def get_links(urls):
    with webdriver.Chrome(service=ChromeService(ChromeDriverManager().install())) as driver:
        sources = []

        for url in urls:
            driver.get(url)
            sources.append(driver.page_source)

    rss = []
    for page in sources:
        rss.append(json.loads(BeautifulSoup(page, 'lxml').get_text()))

    merged_dict = {}
    if len(rss) > 1:
        for i in rss:
            merged_dict = {**merged_dict, **i}

    return merged_dict


def get_page(url):
    page = requests.get(url)
    print(url, page.status_code)
    return page

    
def REST_codes():
    # TODO
    return 1

def clean_page(page):
    tag = re.compile(r'<[^>]+>')
    soup = BeautifulSoup(page.text, 'html.parser').find_all('p')

    paragraphs = []
    for x in soup:
        paragraphs.append(str(x))

    paragraphs = ' '.join(paragraphs)
    clean_text = re.sub(tag, '', paragraphs)

    return clean_text

def filter_page(url):
    if ("arxiv" in url):
        # Make a function for PDFs here
        # TODO
        return True        

def split_document(document, chunk_size=2000):
    chunks = []
    for i in range(0, len(document), chunk_size):
        chunks.append(document[i:i+chunk_size])
    return chunks

def split_sentences(document):
    return document.split(". ")

def embeddings_model():
    return "text-embedding-ada-002"

def gpt_response(model, message):

    return openai.ChatCompletion.create(
                model=model,
                messages=message
            )

def upsert_documents(embeddings, summary, metadata, index):

    # Creating embeddings for each document and preparing for upsert
    upsert_items = []
    
    embedding = [record['embedding'] for record in embeddings]
    # Include the original document text in the metadata
    document_metadata = metadata.copy()
    document_metadata['original_text'] = summary
    upsert_items.append((f"{metadata['pagename']}", embedding[0], document_metadata))

    # Upsert to Pinecone
    index.upsert(upsert_items)

def num_tokens_from_string(page_string, model) -> int:
    """Returns the number of tokens in a text string."""

    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(page_string))
    
    return num_tokens
