from langchain.document_loaders import JSONLoader, DirectoryLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from typing import List

tiktoken.encoding_for_model("gpt-4")


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["id"] = record.get("id")
    metadata["source"] = record.get("source")
    # metadata["tokens"] = record.get("tokens")

    return metadata


def get_docs(folder) -> List[Document]:
    """return a list of documents with all jsonl files from a directory"""
    return DirectoryLoader(
        folder,
        glob="**/*.jsonl",
        show_progress=True,
        loader_cls=JSONLoader,
        loader_kwargs={
            "jq_schema": ".",
            "content_key": "text",
            "metadata_func": metadata_func,
            "text_content": False,
            "json_lines": True,
        },
    ).load()


def tiktoken_len(text):
    # create the length function
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def split_documents(
    docs: List[Document], chunk_size=1000, chunk_overlap=40
) -> List[Document]:
    """split the text according to the chunk size"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,  # number of tokens overlap between chunks
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter.split_documents(docs)
