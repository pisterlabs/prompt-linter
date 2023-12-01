from typing import List
import re
from pydantic import BaseModel
from langchain.document_loaders import UnstructuredFileLoader
from unstructured.cleaners.core import clean_extra_whitespace
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter


def find_files(pattern) -> List[str]:
    """
    Find files matching a pattern
    :param pattern: Pattern to match e.g. 'data/*.txt'
    :return: List of file paths
    """
    import glob

    result: List[str] = []

    for file_path in glob.glob(pattern):
        result.append(file_path)

    return result


def scan_directory(pattern: str,chunk_size=1024, chunk_overlap=100) -> List[List[Document]]:
    """
    Retrieve structured data from a directory
    :param pattern: Pattern to match e.g. 'data/*.txt'
    :return: List of Document objects
    """

    result: List[List[Document]] = []

    for file_path in find_files(pattern):
        result.append(process_file_data(file_path,chunk_overlap=chunk_overlap,chunk_size=chunk_size))

    return result

def scan_urls(urls: List[str], chunk_size=1024, chunk_overlap=100) -> List[List[Document]]:
    """
    Retrieve structured data from a list of URLs
    :param urls: List of URLs
    :return: List of Document objects
    """
    result: List[List[Document]] = []

    for url in urls:
        result.append(process_url_data(url, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    return result

def process_url_data(url: str, chunk_size=1024, chunk_overlap=100) -> List[Document]:
    """
    Retrieve structured data from a URL
    :param url: URL to retrieve
    :return: List of Document objects
    """
    mode="single"

    loader = UnstructuredURLLoader(urls=[url], mode=mode, continue_on_failure=True)

    docs = loader.load()

    if mode == "single":
        return handle_single_text(url, docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    return handle_elements(url, docs)

def process_file_data(file_path: str, chunk_size=1024, chunk_overlap=100) -> List[Document]:
    """
    Retrieve structured data from a file
    :param file_path: Path to the file
    :return: List of Document objects
    """

    mode = "single"

    loader = UnstructuredFileLoader(
        file_path=file_path,
        strategy="hi-res",  # other option:"fast"
        mode=mode,  # single (default), elements, paged (for PDFs)
        post_processors=[clean_extra_whitespace],
    )

    docs = loader.load()

    if mode == "single":
        if file_path.endswith(".md"):
            return handle_single_md(docs, file_path=file_path)

        return handle_single_text(file_path, docs, chunk_size, chunk_overlap)

    return handle_elements(file_path, docs)

def handle_elements(file_path: str, docs: List[Document]) -> List[Document]:
    """
    Handle when UnstructuredFileLoader is in mode=elements
    """
    result: List[Document] = []
    text = []

    for doc in docs:

        # Check if metadata and category exist, otherwise treat content as part of the answer
        category = doc.metadata.get("category") if doc.metadata else None
        content = doc.page_content.strip()

        if category is None:
            result.append(Document(
                page_content=content,
                    metadata=transform_dict_arrays_to_strings(
                        {**doc.metadata, "file": file_path},
                    ),
            ))
            continue

        if category == "Title":
            if len(text) > 0:
                result.append(Document(
                    page_content=content + "\n".join(text),
                    metadata=transform_dict_arrays_to_strings(
                        {**doc.metadata, "file": file_path},
                    ),
                ))

                text = []
        else:
            if category == "ListItem":
                text.append(f"• {content}")
            else:
                text.append(content)

    # The rest
    if len(text) > 0:
        result.append(Document(
            page_content="\n".join(text),
            metadata={**doc.metadata, "file": file_path},
        ))

    return result


def handle_single_md(docs: List[Document], file_path: str) -> List[Document]:
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "Header 1"),  # Level -> metadata key
        ("##", "Header 2"),
        ("###", "Header 3"),
    ])

    splitted_docs: List[Document] = []

    for doc in docs:
        for split in splitter.split_text(doc.page_content):
            splitted_docs.append(Document(
                page_content=split.page_content.strip(),
                metadata=transform_dict_arrays_to_strings({**split.metadata, "file": file_path}),
            ))

    return splitted_docs


def handle_single_text(
        file_path: str,
        docs: List[Document],
        chunk_size: int,
        chunk_overlap: int) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    docs = text_splitter.split_documents(docs)
    result: List[Document] = []

    for doc in docs:
        result.append(Document(
            page_content=doc.page_content.strip(),
            metadata=transform_dict_arrays_to_strings({**doc.metadata, "file": file_path}),
        ))

    return result

def transform_dict_arrays_to_strings(input_dict):
    """
    Transforms any array in the _input_dict_ to a comma-separated string
    """
    for key, value in input_dict.items():
        # Check if the value is a list
        if isinstance(value, list):
            # Join the list elements into a comma-separated string
            input_dict[key] = ', '.join(map(str, value))
    return input_dict

def is_binary_file(file_name):
    # Common binary file extensions
    binary_extensions = {
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff',
        '.pdf', '.zip', '.rar',
        '.7z', '.mp3', '.wav', '.wma', '.mp4', '.mov',
        '.avi', '.flv', '.mkv'
    }

    # Get the file extension
    extension = file_name.lower().rsplit('.', 1)[-1]
    extension = '.' + extension

    # Check if the extension is in the list of binary extensions
    return extension in binary_extensions
