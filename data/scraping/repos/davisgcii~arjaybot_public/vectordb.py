import os
import requests
import openai
import hashlib
import random, string
from dotenv import load_dotenv
from langchain.document_loaders import (
    PyPDFLoader,
    CSVLoader,
)
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

load_dotenv()

# info here: https://github.com/hwchase17/langchain/blob/master/langchain/vectorstores/pinecone.py

openai.api_key = os.environ["OPENAI_API_KEY"]
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_API_ENV = os.environ["PINECONE_API_ENV"]
index_name = os.environ["PINECONE_INDEX_NAME"]
NUM_DOCS_TO_RETRIEVE = 10


embeddings = OpenAIEmbeddings()

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV,  # next to api key in console
)

index = pinecone.Index(index_name)
vectorstore = Pinecone(index, embeddings.embed_query, "text")


import hashlib


def compute_hash(file_content):
    """Computes a hash of the file contents.

    Keyword arguments:
    file_content -- the file contents
    returns -- a sha256 hash of the file contents
    """
    sha256 = hashlib.sha256()
    sha256.update(file_content)
    return sha256.hexdigest()


def is_duplicate(doc_hash):
    """Checks to see if a doc exists in the vectorstore with the same hash value.

    Keyword arguments:
    doc_hash -- the hash of the document
    returns -- a bool, True if the doc exists in the vectorstore, False otherwise
    """
    docs = get_docs("any subject here", filter={"doc_hash": doc_hash})

    if len(docs) > 0:
        return True
    else:
        return False


def get_docs(query, filter=None):
    """Get relevant docs from the vectorstore.

    Keyword arguments:
    query -- the user's query
    returns -- a list of relevant documents
    """
    docs = vectorstore.similarity_search(query, k=NUM_DOCS_TO_RETRIEVE, filter=filter)

    return docs


def add_pages(document, doc_hash, user_id):
    """Adds pages from a document to the vectorstore.

    Keyword arguments:
    document -- the pdf document (a list of Document objects)
    doc_hash -- the hash of the document
    user_id -- the user's id
    returns -- nothing
    """
    flagged = [is_unsafe_content(page.page_content) for page in document]

    if any(flagged):
        return

    pages = [page.page_content for page in document]
    metadatas = [page.metadata for page in document]

    for metadata in metadatas:
        metadata["doc_hash"] = doc_hash

        if user_id is not None:
            metadata["user_id"] = user_id

    vectorstore.add_texts(texts=pages, metadatas=metadatas)


def add_pdf(url, user_id):
    """Add a pdf from a url to the vectorstore.

    Keyword arguments:
    url -- the url of the pdf (can be a private URL if uploaded from slack)
    user -- the slack user's id
    returns -- bool, True if the document is a duplicate, False otherwise
    """
    response = requests.get(
        url, headers={"Authorization": "Bearer %s" % SLACK_BOT_TOKEN}
    )

    pdf_content = response.content
    doc_hash = compute_hash(pdf_content)

    if is_duplicate(doc_hash):
        return True

    # generate a random file name of 32 characters -- so multiple users can upload at the same time
    file_name = "".join(random.choices(string.ascii_letters + string.digits, k=16))
    file_name += ".pdf"

    with open(file_name, "wb") as f:
        f.write(response.content)

    loader = PyPDFLoader(file_name)

    document = loader.load_and_split()

    add_pages(document, doc_hash, user_id)

    os.remove(file_name)

    return False


def add_csv(url, user_id):
    """Add a csv from a url to the vectorstore.

    Keyword arguments:
    url -- the url of the csv (can be a private URL if uploaded from slack)
    user_id -- the slack user's id
    returns -- bool, True if the document is a duplicate, False otherwise
    """
    response = requests.get(
        url, headers={"Authorization": "Bearer %s" % SLACK_BOT_TOKEN}
    )

    csv_content = response.content
    doc_hash = compute_hash(csv_content)
    if is_duplicate(doc_hash):
        return True

    # generate a random file name of 32 characters -- so multiple users can upload at the same time
    file_name = "".join(random.choices(string.ascii_letters + string.digits, k=16))
    file_name += ".csv"

    with open(file_name, "wb") as f:
        f.write(response.content)

    loader = CSVLoader(file_name)

    document = loader.load()

    add_pages(document, doc_hash, user_id)

    os.remove(file_name)

    return False


def is_unsafe_content(text):
    """Checks to see if any text content is unsafe.

    Keyword arguments:
    text -- text from any source, a python string
    returns -- a bool, True if the content is unsafe as flagged by OpenAI
    """
    response = openai.Moderation.create(input="Sample text goes here")

    output = response["results"][0]

    return output["flagged"]
