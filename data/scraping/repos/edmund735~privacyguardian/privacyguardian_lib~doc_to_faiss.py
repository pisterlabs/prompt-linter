from dotenv import load_dotenv

import os

import json

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.llms import OpenAI

load_dotenv()
embeddings = OpenAIEmbeddings()
llm = OpenAI(temperature=0)

with open("data_index.json", "r") as f:
    data_index = json.load(f)

def make_data_path(index_number):
    return os.path.join(
        "data/OPP-115/sanitized_policies/",
        data_index[str(index_number)],
    )


def make_db_path(input_file):
    return os.path.join("db", f"{input_file}.faiss")


def make_faiss_db(input_file):
    raw_documents = TextLoader(input_file).load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separator="<br>"
    )
    documents = text_splitter.split_documents(raw_documents)

    db = FAISS.from_documents(documents, embeddings)
    db.save_local(make_db_path(input_file))
    return db


if __name__ == "__main__":
    for i in data_index.keys():
        p = make_data_path(i)
        make_faiss_db(p)
