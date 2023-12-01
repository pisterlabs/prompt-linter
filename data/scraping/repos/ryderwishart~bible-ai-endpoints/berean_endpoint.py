from pathlib import Path
from modal import Image, Secret, Stub, web_endpoint

image = Image.debian_slim().pip_install(
    # langchain pkgs
    "langchain~=0.0.209",
    "hnswlib==0.7.0",
    "tiktoken==0.3.0",
    "sentence_transformers~=2.2.2",
    "chromadb~=0.3.23",
    "openai~=0.27.8",
    "bs4~=0.0.1",
)
stub = Stub(
    name="bible-chroma",
    image=image,
    secrets=[Secret.from_name("openai-qa")], # Note: you need a secret in your modal.com account named "openai-qa" to use any OpenAI API calls
)

import requests
import zipfile
import io
import logging
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI

# Set up embeddings
embeddings = HuggingFaceEmbeddings()

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

bible_chroma = None

def load_bible_database():
    global bible_chroma

    # Log that we are loading the Bible Chroma database
    logger.info("Loading Bible Chroma database...")

    # Download the Bible Chroma database from Google Drive
    url = "https://github.com/ryderwishart/biblical-machine-learning/raw/main/data/berean-bible-vectorstore.zip"
    response = requests.get(url)

    # Make sure the request was successful
    response.raise_for_status()

    # Create a BytesIO object from the content
    file = io.BytesIO(response.content)

    # Open the zip file
    with zipfile.ZipFile(file, "r") as zip_ref:
        # Extract all the contents of zip file in current directory
        zip_ref.extractall()

    # Load the Bible Chroma database into a Chroma object
    bible_persist_path = "berean-bible-vectorstore"
    bible_chroma = Chroma(
        "berean-bible-vectorstore",
        embeddings, 
        persist_directory=bible_persist_path
    )

    return bible_chroma


# If the directory 'bible' does not exist, then load the bible Chroma database
if not Path("berean-bible-vectorstore").exists():
    load_bible_database()


@stub.function()
@web_endpoint()
def get_documents(query: str, k=5, show_sources: bool = False, mmr=False):
    global bible_chroma
    logger.info("Querying Bible Chroma database...")
    # This method can be called multiple times without reloading the database
    if mmr:
        result = bible_chroma.search(query, k=k, search_type="mmr")
    else:
        result = bible_chroma.search(query, k=k, search_type="similarity")
    logger.info(f"Result: {result}")
    return result


@stub.function()
@web_endpoint(method="GET")
def question(query: str, show_sources: bool = False):
    global bible_chroma
    logger.info("Starting QA Retrieval Chain...")

    llm = ChatOpenAI(
        # model_name="gpt-3.5-turbo-16k",
        model_name="gpt-3.5-turbo",
        temperature=0,
        streaming=True,
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=bible_chroma.as_retriever(kwargs={"k": 1}),
        reduce_k_below_max_tokens=True,
        max_tokens_limit=2000,
    )

    result = chain(query)
    logging.info(f"Result: {result}")

    return result
