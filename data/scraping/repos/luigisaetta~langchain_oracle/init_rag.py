#
# This one is to be used in Notebooks
#

# for pdf post processing
import re

# modified to load from Pdf
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# two possible vector store
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS

from langchain.schema.runnable import RunnablePassthrough

# removed OpenAI, using Cohere embeddings
from langchain.embeddings import CohereEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings


from langchain import hub

import oci

from langchain.llms import Cohere

# oci_llm is in a local file
from oci_llm import OCIGenAILLM

# config for the RAG
from config_rag import (
    BOOK_LIST,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTOR_STORE_NAME,
    MAX_TOKENS,
    ENDPOINT,
    EMBED_TYPE,
    MAX_DOCS_RETRIEVED,
    TEMPERATURE,
    EMBED_HF_MODEL_NAME,
    TIMEOUT,
    LLM_TYPE,
)

# private configs
from config_private import COMPARTMENT_OCID, COHERE_API_KEY

DEBUG = False

CONFIG_PROFILE = "DEFAULT"


#
# def load_oci_config()
#
def load_oci_config():
    # read OCI config to connect to OCI with API key
    oci_config = oci.config.from_file("~/.oci/config", CONFIG_PROFILE)

    # check the config to access to api keys
    if DEBUG:
        print(oci_config)

    return oci_config


#
# do some post processing on text
#
def post_process(splits):
    for split in splits:
        split.page_content = split.page_content.replace("\n", " ")
        split.page_content = re.sub("[^a-zA-Z0-9 \n\.]", " ", split.page_content)
        # remove duplicate blank
        split.page_content = " ".join(split.page_content.split())

    return splits


#
# def: Initialize_rag_chain
#
def initialize_rag_chain():
    # Initialize RAG

    # Loading a list of pdf documents
    all_pages = []

    # modified to load a list of pdf
    for book in BOOK_LIST:
        print(f"Loading book: {book}...")
        loader = PyPDFLoader(book)

        # loader split in pages
        pages = loader.load()
        print(f"Loaded {len(pages)} pages...")

        all_pages.extend(pages)

        print("PDF document loaded!")

    # This split in chunks
    # try with smaller chuncks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    splits = text_splitter.split_documents(all_pages)

    print(f"We have splitted the pdf in {len(splits)} splits...")

    # some post processing
    splits = post_process(splits)

    print("Initializing vector store...")

    if EMBED_TYPE == "COHERE":
        print("Loading Cohere Embeddings Model...")
        embed_model = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)
    if EMBED_TYPE == "LOCAL":
        print(f"Loading HF Embeddings Model: {EMBED_HF_MODEL_NAME}")

        model_kwargs = {"device": "cpu"}
        # changed to True for BAAI, to use cosine similarity
        encode_kwargs = {"normalize_embeddings": True}

        embed_model = HuggingFaceEmbeddings(
            model_name=EMBED_HF_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    # using Chroma as Vector store
    print(f"Indexing: using {VECTOR_STORE_NAME} as Vector Store...")

    if VECTOR_STORE_NAME == "CHROME":
        vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)
    if VECTOR_STORE_NAME == "FAISS":
        vectorstore = FAISS.from_documents(documents=splits, embedding=embed_model)

    # increased num. of docs to 5 (default to 4)
    retriever = vectorstore.as_retriever(search_kwargs={"k": MAX_DOCS_RETRIEVED})

    # Build the class for OCI GenAI

    # Only needed for OCI LLM
    print(f"Using {LLM_TYPE} llm...")

    if LLM_TYPE == "OCI":
        oci_config = load_oci_config()

        llm = OCIGenAILLM(
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            config=oci_config,
            compartment_id=COMPARTMENT_OCID,
            endpoint=ENDPOINT,
            debug=DEBUG,
            timeout=TIMEOUT,
        )
    if LLM_TYPE == "COHERE":
        llm = Cohere(
            cohere_api_key=COHERE_API_KEY,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

    # for now hard coded...
    rag_prompt = hub.pull("rlm/rag-prompt")

    print("Building rag_chain...")
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} | rag_prompt | llm
    )

    print("Init RAG complete...")
    return rag_chain


#
# def: get_answer  from LLM
#
def get_answer(rag_chain, question):
    response = rag_chain.invoke(question)

    print(f"Question: {question}")
    print("The response:")
    print(response)
    print()

    return response
