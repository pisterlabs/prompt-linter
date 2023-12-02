import os
import zlib

import openai
import sys

from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

from llm_experiments.utils import here

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']


# load notion doc from file and unzip.
def get_file_and_unzip(file_path):
    with open(file_path, 'rb') as f:
        file_bytes = f.read()
    unzipped_file_bytes = zlib.decompress(file_bytes)
    return unzipped_file_bytes

fp = here() / 'data/09666167-a2a5-4f6b-91aa-51c1694f2292_Export-1a52ad77-0631-4e05-9eb4-f7221723021e.zip'


import zipfile

import zipfile
from io import BytesIO

import zipfile
import io


def extract_all_text_from_zip(zip_file_path):
    text_data = {}
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            with zip_ref.open(file_info) as file:
                text_data[file_info.filename] = file.read().decode('utf-8')  # Decode binary data as text
    return text_data

# Example usage
text_data = extract_all_text_from_zip(fp)
# Example usage


fp = here() / 'data/Notion_DB/'
loader = NotionDirectoryLoader(fp)
docs = loader.load()
txt = ' '.join([d.page_content for d in docs])

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

md_header_splits = markdown_splitter.split_text(txt)
