# -*- coding: utf-8 -*-
import os
import pickle

import faiss
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import BSHTMLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from lcserve import serving

data_folder = "data"
cache_folder = "scraping_cache"
docs_name = "unrealdocs"
if not os.path.exists(os.path.join(data_folder, cache_folder)):
    os.makedirs(os.path.join(data_folder, cache_folder))

global store


def progress_bar(current, total, bar_length=20):
    """
    Function that creates a simple progress bar.

    Parameters:
        current (int): The current progress.
        total (int): The total progress.
        bar_length (int): The length of the progress bar.

    Returns:
        str: A string representing the progress bar.
    """

    filled_length = int(round(bar_length * current / float(total)))
    remaining_length = bar_length - filled_length
    bar = '█' * filled_length + '-' * remaining_length
    percentage = round(100.0 * current / float(total), 1)
    return f'[{bar}] {percentage}%'


"""## Document Loader
This is a simple document loader that uses BeautifulSoup to extract the text from the HTML.
"""


# def prepare():
#     global store

@serving
def ask(input: str) -> str:
    """## Plain QA Chain"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Use the following pieces of context to answer the Unreal Engine game development related question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
        {context}
    
        Question: {question}
    
        Helpful Answer: """
    )
    chain = load_qa_chain(OpenAI(), chain_type="stuff", prompt=prompt)  # we are going to stuff all the docs in at once

    query = input
    docs = store.similarity_search(query)

    return chain.run(input_documents=docs, question=query)


if not os.environ.get("OPENAI_API_KEY"):
    raise Exception("OPENAI_API_KEY is not set")

if not os.path.exists(os.path.join(data_folder, docs_name + ".index")) or os.path.getsize(os.path.join(data_folder, docs_name + ".index")) == 0:
    print("Creating new index")
    url = "https://docs.unrealengine.com/5.1/en-US/navTree.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all the li tags that do not have the "landing" class.
    # URLs without any content (i.e landing pages) are not useful for us.
    li_tags = soup.select('li:not(.landing) > p > a#\\ sidebar_link[href^="/en-US/"]')

    URLs = []
    for li in li_tags:
        href = li.get('href')
        URLs.append(href)

    # remove duplicates
    URLs = list(dict.fromkeys(URLs))
    print("Number of URLs: " + str(len(URLs)))

    """## Scraping the docs
    We are going to scrape the docs and store the raw text in a variable and then cache it to a file after we remove the newlines.
    """
    raw_text = ""
    # if size of cache is 0, then we need to scrape the docs
    if os.path.exists(os.path.join(data_folder, cache_folder, docs_name + ".raw")) and os.path.getsize(
            os.path.join(data_folder, cache_folder, docs_name + ".raw")) > 0:
        print("Loading raw text from cache")
        file = open(os.path.join(data_folder, cache_folder, docs_name + ".raw"), "r", encoding="utf-8")
        raw_text = file.read()
        file.close()
        print("Loaded raw text from: " + os.path.join(data_folder, cache_folder, docs_name + ".raw"))
    else:
        print("Scraping the docs")
        progressPercent = 0
        total = len(URLs)
        for index, url in enumerate(URLs):
            url = "https://docs.unrealengine.com" + url
            r = requests.get(url, allow_redirects=True)
            soup = BeautifulSoup(r.content, 'html.parser')
            content = soup.get_text()
            raw_text += content
            progressPercent += 1
            print(progress_bar(progressPercent, total), end="\r")

        print("Removing newlines")
        index = 20
        while index > 1:  # we dont want to replace single newlines
            raw_text = raw_text.replace("\n" * index, "\n")
            index -= 1

        # Caching the raw text
        file = open(os.path.join(data_folder, cache_folder, docs_name + ".raw"), "w+", encoding="utf-8")
        file.write(raw_text)
        file.close()
        print("Saved raw text to: " + os.path.join(data_folder, cache_folder, docs_name + ".raw" + "clean"))

    print("Number of characters: " + str(len(raw_text)))
    print("Splitting the text into chunks", end="\r")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,
        chunk_overlap=500,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    print(f"Number of chunks: {len(texts)}", end="\r")

    """## Making the embeddings """
    print("Creating embeddings", end="\r")
    embeddings = OpenAIEmbeddings()

    """## Vector Store
    This is where we store the vectors. We are using FAISS here, but you can use any vector store you want.
    """
    print("Creating vector store")
    store = FAISS.from_texts(texts, embeddings)

    faiss.write_index(store.index, os.path.join(data_folder, docs_name + ".index"))
    print("Saved vector store to: " + os.path.join(data_folder, docs_name + ".index"))
    store.index = None
    with open(os.path.join(data_folder, docs_name + ".pkl"), "wb") as f:
        pickle.dump(store, f)
        print("Saved vector store to: " + os.path.join(data_folder, docs_name + ".pkl"))
else:
    print("Loading from file: " + os.path.join(data_folder, docs_name + ".index"))
    index = faiss.read_index(os.path.join(data_folder, docs_name + ".index"))
    with open(os.path.join(data_folder, docs_name + ".pkl"), "rb") as f:
        store = pickle.load(f)
    store.index = index

print(ask("Respond with the word 'welcome'"))
