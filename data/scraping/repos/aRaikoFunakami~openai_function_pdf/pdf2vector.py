'''
Save vectorstore DB of PDF files specified in "pdf_files" to the location specified in "persist_directory"
'''
import os
import sys
import logging
import json
import openai
from urllib.parse import urlparse

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# load pdf files
pdf_files = [
    "https://manual.lexus.jp/pdf/nx/NX350-NX250_UG_JP_M78364_1_2303.pdf",
    "https://manual.lexus.jp/pdf/nx/NX350-NX250_OM_JP_M78364V_1_2303.pdf",
    "https://manual.lexus.jp/pdf/nx/NX350-NX250_MM_JP_M78364N_1_2303.pdf",
]

# openapi model
model_name = "gpt-3.5-turbo-0613"

# how to split
use_text_splitter = False
persist_directory = None
if use_text_splitter:
    persist_directory = "./chroma_split_documents"
else:
    persist_directory = "./chroma_load_and_split"


def load_config():
    args = sys.argv
    config_file = (
        os.path.dirname(__file__) + "/config.json" if len(args) <= 1 else args[1]
    )
    logging.info(config_file)
    with open(config_file, "r") as file:
        config = json.load(file)
    return {
        "openai_api_key": config["openai_api_key"],
    }


def is_url(input_string):
    parsed_url = urlparse(input_string)
    return parsed_url.scheme != "" and parsed_url.netloc != ""


# Preprocessing for using Open　AI
def save_pdf_to_vectorstore(pdf_files):
    # init openai api
    config = load_config()
    openai.api_key = config["openai_api_key"]
    os.environ["OPENAI_API_KEY"] = openai.api_key
    llm = ChatOpenAI(temperature=0, model_name=model_name)
    embeddings = OpenAIEmbeddings()
    # splitting pdf to
    pages = []
    for file in pdf_files:
        if is_url(file):
            pdf_file = file
        else:
            pdf_file = os.path.dirname(__file__) + f"/templates/{file}"
        print(pdf_file)
        loader = PyPDFLoader(pdf_file)
        if use_text_splitter:
            documents = loader.load_and_split()
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            pages = pages + text_splitter.split_documents(documents)
        else:
            pages = pages + loader.load_and_split()
        print(len(pages))

    vectorstore = Chroma.from_documents(
        pages, embedding=embeddings, persist_directory=persist_directory
    )
    vectorstore.persist()


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(filename)s:%(funcName)s[%(lineno)d] - %(message)s",
    )
    response = save_pdf_to_vectorstore(pdf_files)


if __name__ == "__main__":
    main()
