from pathlib import Path
from modal import Image, Secret, Stub, web_endpoint

image = Image.debian_slim().pip_install(
    "lxml~=4.9.2",
    # langchain pkgs
    "langchain~=0.0.209",
    "tiktoken==0.3.0",
    "sentence_transformers~=2.2.2",
    "chromadb~=0.3.23",
    "openai~=0.27.8",
    "bs4~=0.0.1",
)
stub = Stub(
    name="tyndale-chroma",
    image=image,
    secrets=[Secret.from_name("openai-qa")],
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


def load_tyndale_database():
    global tyndale_chroma

    # Log that we are loading the Tyndale Chroma database
    logger.info("Loading Tyndale Chroma database...")

    # Download the Tyndale Chroma database from Google Drive
    url = "https://github.com/ryderwishart/biblical-machine-learning/raw/main/data/tyndale.zip"
    response = requests.get(url)

    # Make sure the request was successful
    response.raise_for_status()

    # Create a BytesIO object from the content
    file = io.BytesIO(response.content)

    # Open the zip file
    with zipfile.ZipFile(file, "r") as zip_ref:
        # Extract all the contents of zip file in current directory
        zip_ref.extractall()

    # Load the Tyndale Chroma database into a Chroma object
    tyndale_persist_path = "tyndale"
    tyndale_chroma = Chroma(
        "tyndale", embeddings, persist_directory=tyndale_persist_path
    )

    return tyndale_chroma


# If the directory 'tyndale' does not exist, then load the Tyndale Chroma database
if not Path("tyndale").exists():
    load_tyndale_database()


@stub.function()
@web_endpoint(method="GET")
def get_documents(query: str, k=1, show_sources: bool = False):
    global tyndale_chroma
    logger.info("Querying Tyndale Chroma database...")
    # This method can be called multiple times without reloading the database
    result = tyndale_chroma.similarity_search(query, k=k)
    logger.info(f"Result: {result}")
    return result


@stub.function()
@web_endpoint(method="GET")
def question(query: str, show_sources: bool = False):
    global tyndale_chroma
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
        retriever=tyndale_chroma.as_retriever(kwargs={"k": 1}),
        reduce_k_below_max_tokens=True,
        max_tokens_limit=2000,
    )

    result = chain(query)
    logging.info(f"Result: {result}")

    return result
