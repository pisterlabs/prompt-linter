import sys
import pathlib

# add helpers folder to path (required for Streamlit to find the helpers module)
sys.path.append(str(pathlib.Path().absolute()) + "/helpers")


import feedparser
import os
import pinecone
import openai
import requests
from bs4 import BeautifulSoup
from retrying import retry
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import hashlib
import streamlit as st
import urllib.parse
from helpers import tiktoken_len, create_embedding, crawl, scrape_website

import dotenv

dotenv.load_dotenv(dotenv_path='../.env')


# check environment variables
if os.getenv('PINECONE_API_KEY') is None:
    st.error("PINECONE_API_KEY not set. Please set this environment variable and restart the app.")
    st.stop()
if os.getenv('PINECONE_ENVIRONMENT') is None:
    st.error("PINECONE_ENVIRONMENT not set. Please set this environment variable and restart the app.")
    st.stop()
if os.getenv('OPENAI_API_KEY') is None:
    st.error("OPENAI_API_KEY not set. Please set this environment variable and restart the app.")
    st.stop()

# app starts here
st.title("Upload content to Pinecone 🔎")

st.write("Click Upload to upload baeke.info or other posts to Pinecone in chunks.")

url = st.text_input("Address", "https://blog.baeke.info/feed/")

address_type = st.selectbox("Address type", ["RSS", "Crawl"])

urls = []
if address_type == "RSS":
    # Parse the RSS feed with feedparser
    st.write("Parsing RSS feed: ", url)

    try:
        urls = feedparser.parse(url)
    except Exception as e:
        st.exception(e)
        st.stop()

    # store all the entries in a pages list and display
    # number of pages
    pages = urls.entries
    num_pages = len(pages)
    if num_pages == 0:
        st.write("Error processing feed. Stopping...")
        st.stop()
    st.write("Number of entries: ", num_pages)
elif address_type == "Crawl":
    # fill entries with all links until a certain depth
    pages = scrape_website(url, 2)
    num_pages = len(pages)
    if num_pages == 0:
        st.write("Error processing feed. Stopping...")
        st.stop()
    st.write("Number of entries: ", num_pages)
    print(pages)

with st.expander("Options", expanded=False):
    chunk_size = st.slider("Chunk size", 100, 600, 400)
    chunk_overlap = st.slider("Chunk overlap", 0, 60, 20)
    blog_entries = st.slider("Blog entries", 1, num_pages, num_pages)
    recreate = st.checkbox("Recreate index", False)

if st.button("Upload"):
    # OpenAI API key
    openai.api_key = os.getenv('OPENAI_API_KEY')

    # get the Pinecone API key and environment
    pinecone_api = os.getenv('PINECONE_API_KEY')
    pinecone_env = os.getenv('PINECONE_ENVIRONMENT')

    pinecone.init(api_key=pinecone_api, environment=pinecone_env)

    if "blog-index" not in pinecone.list_indexes():
        st.write("Index does not exist. Creating...")
        pinecone.create_index("blog-index", 1536, metadata_config= {"indexed": ["url", "chunk-id"]})
    else:
        st.write("Index already exists.")
        if recreate:
            st.write("Deleting existing index...")
            pinecone.delete_index("blog-index")
            st.write("Creating new index...")
            pinecone.create_index("blog-index", 1536, metadata_config= {"indexed": ["url", "chunk-id"]})
        else:
            st.write("Reusing existing index.")

    # set index; must exist
    index = pinecone.Index('blog-index')

    # create recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,  # number of tokens overlap between chunks
        length_function=tiktoken_len,
        separators=['\n\n', '\n', ' ', '']
    )

    # starting the upload process
    progress_text = "Upload in progress..."
    my_bar = st.progress(0, text=progress_text)

    pinecone_vectors = []

    with st.expander("Logs", expanded=False):
        for i, entry in enumerate(pages[:blog_entries]):
            page = entry['link']
            r = requests.get(page)
            soup = BeautifulSoup(r.text, 'html.parser')
            if url == "https://blog.baeke.info/feed/":
                article = soup.find("div", {"class": "entry-content"}).text
            else:
                article = soup.text

            st.write("Processing URL: ", page)

            # create chunks
            chunks = text_splitter.split_text(article)

            # create md5 hash of page
            url = page
            url_hash = hashlib.md5(url.encode("utf-8"))       
            url_hash = url_hash.hexdigest()

            number_of_chunks = len(chunks)

            # create embeddings for each chunk
            for j, chunk in enumerate(chunks):
                st.write("\tCreating embedding for chunk ", j+1, " of ", number_of_chunks)
                vector = create_embedding(chunk)

                # concatenate hash and j
                hash_j = url_hash + str(j)

                # add vector to pinecone_vectors list
                st.write("\tAdding vector to pinecone_vectors list for chunk ", j+1, " of ", number_of_chunks)
                pinecone_vectors.append((hash_j, vector, {"url": page, "chunk-id": j, "text": chunk}))

                # upsert every 100 vectors
                if len(pinecone_vectors) % 100 == 0:
                    st.write("Upserting batch of 100 vectors...")
                    upsert_response = index.upsert(vectors=pinecone_vectors)
                    pinecone_vectors = []
            
            # update progress bar
            my_bar.progress((i+1)/blog_entries, text=progress_text + f" {i+1} of {blog_entries}")

    # if there are any vectors left, upsert them
    if len(pinecone_vectors) > 0:
        upsert_response = index.upsert(vectors=pinecone_vectors)
        pinecone_vectors = []

    my_bar.progress(1.0, text="Upload complete.")
