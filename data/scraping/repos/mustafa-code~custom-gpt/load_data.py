import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from consts import CHROMA_SETTINGS, DOCUMENT_MAP, PERSIST_DIRECTORY, INGEST_THREADS, SOURCE_DIRECTORY

def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    file_extension = os.path.splitext(file_path)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined")
    return loader.load()[0]


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        data_list = [future.result() for future in futures]
        # return data and file paths
        return (data_list, filepaths)


def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory
    all_files = os.listdir(source_dir)
    paths = []
    for file_path in all_files:
        file_extension = os.path.splitext(file_path)[1]
        source_file_path = os.path.join(source_dir, file_path)
        if file_extension in DOCUMENT_MAP.keys():
            paths.append(source_file_path)

    return get_docs(paths)

def get_docs(paths):
    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            future = executor.submit(load_document_batch, filepaths)
            futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            contents, _ = future.result()
            docs.extend(contents)

    return docs


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs = []
    for doc in documents:
        text_docs.append(doc)

    return text_docs


def main():
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents = split_documents(documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(text_documents)

    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()
    db = None


if __name__ == '__main__':
    main()
