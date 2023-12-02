# This file is for configuration of the langchain package.
# The configuration is loaded from the langchain.config module.
#
# The configuration is a dictionary with the following keys:
#
# MODEL_TYPE: The type of model to use.
# MODEL: The name of the model to use.
# MODEL_PATH: The path to the model to use.
# EMBEDDINGS_MODEL_NAME: The name of the embeddings model to use.
# MODEL_N_CTX: The number of tokens to process
# TARGET_SOURCE_CHUNKS: The number of chunks to split the source into
# INGEST_THREADS: The number of threads to use for ingestion
# SOURCE_DIRECTORY: The folder to store the source documents
# PERSIST_DIRECTORY: The folder to store
# LOADER_MAP: A dictionary mapping file extensions to Document Loader classes

import os
# Import the Document Loaders
from langchain.document_loaders import (
        CSVLoader,
        PDFMinerLoader,
        TextLoader,
        EverNoteLoader,
        UnstructuredExcelLoader,
        UnstructuredEPubLoader,
        UnstructuredHTMLLoader,
        UnstructuredMarkdownLoader,
        UnstructuredODTLoader,
        UnstructuredPowerPointLoader,
        UnstructuredWordDocumentLoader,
)

# Set model name and path
MODEL_TYPE='Cohere'
API_KEY='n0oQcCiDDdduOa5R6p9f7Sluadpp5ckcIPJ7mMb3'
MODEL='ggml-gpt4all-j-v1.3-groovy.bin'
MODEL_PATH=f'models/{MODEL}'
EMBEDDINGS_MODEL_NAME='all-MiniLM-L6-v2'
MODEL_N_CTX=1000
TARGET_SOURCE_CHUNKS=4

# Set Voice model
VOICE_REC_ENGINE='SpeechRecognition'
VOICE_ENGINE='pyttsx3'

#define root directory
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
# Define the folder for storing database
SOURCE_DIRECTORY = os.path.join(ROOT_DIRECTORY,"source_documents")
PERSIST_DIRECTORY = os.path.join(ROOT_DIRECTORY,"db")

# Change this to the number of threads you want to use for ingestion
INGEST_THREADS = os.cpu_count() or 8

# Map the Document Loader to its file extension
LOADER_MAP = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".xls": (UnstructuredExcelLoader,{}),
    ".xlsx": (UnstructuredExcelLoader,{}),
}
