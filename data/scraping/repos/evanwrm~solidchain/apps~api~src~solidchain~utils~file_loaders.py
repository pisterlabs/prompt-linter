import tempfile
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from fastapi import HTTPException, UploadFile
from langchain.document_loaders import (
    AZLyricsLoader,
    CollegeConfidentialLoader,
    GutenbergLoader,
    HNLoader,
    IMSDbLoader,
    OnlinePDFLoader,
    UnstructuredFileLoader,
    UnstructuredURLLoader,
    YoutubeLoader,
)
from langchain.text_splitter import CharacterTextSplitter

from solidchain.schemas.vectorstore import VectorStoreDB


def from_file(file: UploadFile):
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(file.file.read())

        # Note: depending on filetype, files could contain malicious contents
        # TODO: Attempt to safely extract text from files
        loader = UnstructuredFileLoader(temp_file.name)

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = loader.load_and_split(text_splitter)

    return documents


def from_url(url: str):
    parsed_url = urlparse(url)

    # Domain specific loaders
    match parsed_url.hostname:
        case "www.azlyrics.com" | "azlyrics.com":
            loader = AZLyricsLoader(url)
        case "www.collegeconfidential.com" | "collegeconfidential.com":
            loader = CollegeConfidentialLoader(url)
        case "www.gutenberg.org" | "gutenberg.org":
            loader = GutenbergLoader(url)
        case "www.imsdb.com" | "imsdb.com":
            loader = IMSDbLoader(url)
        case "news.ycombinator.com":
            loader = HNLoader(url)
        case "www.youtube.com" | "youtube.com" | "youtu.be":
            loader = YoutubeLoader(url)
    # Generic loaders
    if loader is None:
        match parsed_url.path.split(".")[-1]:
            case "pdf":
                loader = OnlinePDFLoader(url)
            case _:
                loader = UnstructuredURLLoader(url)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = loader.load_and_split(text_splitter)

    return documents
