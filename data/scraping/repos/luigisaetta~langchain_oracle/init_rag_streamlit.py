#
# This one is to be used with Streamlit
#

import streamlit as st

# for pdf post processing
import re

# modified to load from Pdf
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# for caching
from langchain.storage import LocalFileStore

# two possible vector store
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS

from langchain.schema.runnable import RunnablePassthrough

# removed OpenAI, using Cohere embeddings
from langchain.embeddings import CohereEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import CacheBackedEmbeddings

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

from langchain import hub
from langchain.prompts import ChatPromptTemplate

from langchain.llms import Cohere

import oci

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
    EMBED_COHERE_MODEL_NAME,
    MAX_DOCS_RETRIEVED,
    ADD_RERANKER,
    TEMPERATURE,
    EMBED_HF_MODEL_NAME,
    TIMEOUT,
    LLM_TYPE,
    DEBUG
)

# private configs
from config_private import COMPARTMENT_OCID, COHERE_API_KEY

CONFIG_PROFILE = "DEFAULT"


#
# load_oci_config(): load the OCI security config
#
def load_oci_config():
    # read OCI config to connect to OCI with API key
    oci_config = oci.config.from_file("~/.oci/config", CONFIG_PROFILE)

    # check the config to access to api keys
    if DEBUG:
        print()
        print("OCI Config:")
        print(oci_config)
        print()

    return oci_config


#
# do some post processing on text
#
def post_process(splits):
    for split in splits:
        # replace newline with blank
        split.page_content = split.page_content.replace("\n", " ")
        split.page_content = re.sub("[^a-zA-Z0-9 \n\.]", " ", split.page_content)
        # remove duplicate blank
        split.page_content = " ".join(split.page_content.split())

    return splits


#
# load all pages from pdf books
#
#
def load_all_pages(book_list):
    all_pages = []

    for book in book_list:
        print(f"Loading book: {book}...")
        loader = PyPDFLoader(book)

        # loader split in pages
        pages = loader.load()

        all_pages.extend(pages)

        print(f"Loaded {len(pages)} pages...")

    return all_pages


#
# Split pages in chunk
#
def split_in_chunks(all_pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    splits = text_splitter.split_documents(all_pages)

    # some post processing on text
    splits = post_process(splits)

    print(f"Splitted the pdf in {len(splits)} chunks...")

    return splits


#
# Load the embedding model
#
def create_cached_embedder():
    print("Initializing Embeddings model...")

    # Introduced to cache embeddings and make it faster
    fs = LocalFileStore("./vector-cache/")

    if EMBED_TYPE == "COHERE":
        print("Loading Cohere Embeddings Model...")
        embed_model = CohereEmbeddings(
            model=EMBED_COHERE_MODEL_NAME, cohere_api_key=COHERE_API_KEY
        )
    elif EMBED_TYPE == "LOCAL":
        print(f"Loading HF Embeddings Model: {EMBED_HF_MODEL_NAME}")

        model_kwargs = {"device": "cpu"}
        # changed to True for BAAI, to use cosine similarity
        encode_kwargs = {"normalize_embeddings": True}

        embed_model = HuggingFaceEmbeddings(
            model_name=EMBED_HF_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    # the cache for embeddings
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embed_model, fs, namespace=embed_model.model_name
    )

    return cached_embedder


#
# create retrievere with optional reranker
#
def create_retriever(vectorstore):
    if ADD_RERANKER == False:
        # no reranking
        print("No reranking...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": MAX_DOCS_RETRIEVED})
    else:
        # to add reranking
        print("Adding reranking to QA chain...")

        compressor = CohereRerank(cohere_api_key=COHERE_API_KEY)

        base_retriever = vectorstore.as_retriever(
            search_kwargs={"k": MAX_DOCS_RETRIEVED}
        )

        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )

    return retriever


#
# create vector store
#
def create_vector_store(store_type, document_splits, embedder):
    print(f"Indexing: using {store_type} as Vector Store...")

    if store_type == "CHROME":
        # modified to cache
        vectorstore = Chroma.from_documents(
            documents=document_splits, embedding=embedder
        )
    elif store_type == "FAISS":
        # modified to cache
        vectorstore = FAISS.from_documents(
            documents=document_splits, embedding=embedder
        )

    return vectorstore


#
# Build LLM
#
def build_llm(llm_type):
    print(f"Using {llm_type} llm...")

    if llm_type == "OCI":
        oci_config = load_oci_config()

        llm = OCIGenAILLM(
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            config=oci_config,
            compartment_id=COMPARTMENT_OCID,
            service_endpoint=ENDPOINT,
            debug=DEBUG,
            timeout=TIMEOUT,
        )
    elif llm_type == "COHERE":
        llm = Cohere(
            model="command",  # using large model and not nightly
            cohere_api_key=COHERE_API_KEY,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

    return llm


#
# Initialize_rag_chain
#
# to run it only once
@st.cache_resource
def initialize_rag_chain():
    # Initialize RAG

    # 1. Load a list of pdf documents
    all_pages = load_all_pages(BOOK_LIST)

    # 2. Split pages in chunks
    document_splits = split_in_chunks(all_pages)

    # 3. Load embeddings model
    embedder = create_cached_embedder()

    # 4. Create a Vectore Store and store embeddings
    vectorstore = create_vector_store(VECTOR_STORE_NAME, document_splits, embedder)

    # 5. Create a retriever
    # increased num. of docs to 5 (default to 4)
    # added optionally a reranker
    retriever = create_retriever(vectorstore)

    # 6. Build the LLM
    llm = build_llm(LLM_TYPE)

    # 7. define the prompt (for now hard coded...)
    # rag_prompt = hub.pull("rlm/rag-prompt")
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    rag_prompt = ChatPromptTemplate.from_template(template)

    # 8. build the entire RAG chain
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

    if DEBUG:
        print(f"Question: {question}")
        print("The response:")
        print(response)
        print()

    return response
