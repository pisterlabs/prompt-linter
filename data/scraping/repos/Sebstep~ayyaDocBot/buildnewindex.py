import openai
from dotenv import load_dotenv

import os, argparse
from dotenv import load_dotenv
import openai
from llama_index import (
    StorageContext,
    ServiceContext,
    set_global_service_context,
    get_response_synthesizer,
    load_index_from_storage,
    SimpleDirectoryReader,
)
from llama_index.llms import OpenAI
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from storageLogistics import build_new_storage

# define LLM

# setup
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
STORAGE_FOLDER = os.getenv("STORAGE_FOLDER")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER")
STORAGE_TYPE = os.getenv("STORAGE_TYPE")

DOC_LIMIT = False  # set to integer for testing

if DOC_LIMIT:
    documents = SimpleDirectoryReader(
        "documents/new", filename_as_id=True, num_files_limit=DOC_LIMIT
    ).load_data()
else:
    documents = SimpleDirectoryReader("documents/new", filename_as_id=True).load_data()


build_new_storage(documents, type=STORAGE_TYPE, storage_folder=STORAGE_FOLDER)
