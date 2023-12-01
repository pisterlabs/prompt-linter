import os
import time
from pathlib import Path


from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from typing import List
import ray
# from embeddings import LocalHuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup
import requests


# To download the files locally for processing, here's the command line
# wget -e robots=off --recursive --no-clobber --page-requisites --html-extension \
# --convert-links --restrict-file-names=windows \
# --domains docs.ray.io --no-parent https://docs.ray.io/en/master/

FAISS_INDEX_PATH="faiss_index"

# loader = ReadTheDocsLoader("docs.ray.io/en/master/")

# We'll save our headlines to this path
file_path = Path("transcript.txt")

# Download headlines from NYT
def download_headlines():
    res = requests.get("https://www.nytimes.com")
    soup = BeautifulSoup(res.content, "html.parser")
    # Grab all headlines
    headlines = soup.find_all("h3", class_="indicate-hover", string=True)
    parsed_headlines = []
    for h in headlines:
        parsed_headlines.append(h.get_text())

    # Write headlines to a text file
    with open(file_path, "w") as f:
        f.write(str(parsed_headlines))
        f.close()

if not file_path.exists():
    download_headlines()
    

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 300,
    chunk_overlap  = 20,
    length_function = len,
)

# Stage one: read all the docs, split them into chunks. 
st = time.time() 
print('Loading documents ...')
# docs = loader.load()
#Theoretically, we could use Ray to accelerate this, but it's fast enough as is. 
with open(file_path) as f:
    saved_file = f.read()
    chunks = text_splitter.split_text(saved_file)
# chunks = text_splitter.create_documents([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])
print(chunks)
# raise ValueError
et = time.time() - st
print(f'Time taken: {et} seconds.') 

#Stage two: embed the docs. 
# embeddings = LocalHuggingFaceEmbeddings('multi-qa-mpnet-base-dot-v1')
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# print(embeddings)
print(f'Loading chunks into vector store ...') 
st = time.time()
# print(embeddings[0])
db = FAISS.from_texts(chunks, embeddings)
db.save_local(FAISS_INDEX_PATH)
et = time.time() - st
print(f'Time taken: {et} seconds.')