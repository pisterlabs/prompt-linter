import os, sys, argparse
from dotenv import load_dotenv
import openai
from llama_index import (
    StorageContext,
    ServiceContext,
    set_global_service_context,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.llms import OpenAI
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from storageLogistics import build_new_storage


# setup
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


print("Loading existing storage...")
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)


# create a list of all filenames within the documents/processed directory
my_filenames = os.listdir("documents/processed")

import csv

# save my_filenames to a file
with open("documents/my_filenames.csv", "w", encoding="UTF8") as f:
    writer = csv.writer(f)
    writer.writerow("filename")
    for filename in my_filenames:
        f.write(filename + "\n")
