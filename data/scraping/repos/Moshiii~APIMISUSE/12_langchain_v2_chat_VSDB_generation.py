
from dotenv import load_dotenv
import os
import json
import openai
from chromadb.config import Settings
from chromadb.utils import embedding_functions

import chromadb
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_embedding_vectors():
    with open("C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\misuse_v2_agree.json", encoding="utf-8") as f:
        data = f.readlines()
        data = [line for line in data if line != "\n"]
        data = [json.loads(line) for line in data]
        # print(data)
    documents = []
    ids = []
    for idx in range(len(data)):
        manual_comments = data[idx]["manual_comments"]
        change = data[idx]["change"]
        number = data[idx]["number"]
        added = data[idx]["added"]
        removed = data[idx]["removed"]
        manual_label = data[idx]["manual_label"]
        doc = ""
        doc += "manual_comments: \n"
        doc += manual_comments
        doc += "change: \n"
        doc += change
        doc += "added: \n"
        doc += added
        doc += "removed: \n"
        doc += removed
        doc += "manual_label: \n"
        doc += manual_label
        documents.append(doc)
        ids.append(str(number))
    # print(documents)
    # print(ids)
    # print(len(documents))
    collection = get_VDB()
    collection.add(
        documents=documents,
        ids=ids
    )


def get_VDB():

    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        # Optional, defaults to .chromadb/ in the current directory
        persist_directory="C:\\@code\\APIMISUSE\\data\\embedding\\data_API_fix_categorizor"

    ))
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ['OPENAI_API_KEY'],
        model_name="text-embedding-ada-002"
    )

    collection = client.get_or_create_collection(
        "langchain", embedding_function=openai_ef)

    return collection

# only run it for one time
generate_embedding_vectors()

