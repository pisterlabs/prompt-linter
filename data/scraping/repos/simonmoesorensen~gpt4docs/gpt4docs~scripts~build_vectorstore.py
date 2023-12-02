from typing import Generator
import time
import uuid
from pathlib import Path
import shutil
import logging

from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders.parsers.txt import TextParser
from langchain.document_loaders.blob_loaders import Blob

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format=(
        "[%(asctime)s][%(levelname)s][%(name)s]"
        + "[%(funcName)s) %(filename)s:%(lineno)d] %(message)s"
    ),
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

root = Path(__file__).parent.parent.parent


def delete_existing_vectorstore(directory: Path):
    if directory.exists():
        shutil.rmtree(directory)
        logger.info(f"Deleted existing vectorstore at {directory}")


def load_documents_from_folder(
    folder: Path, docs_per_iter: int
) -> Generator[Blob, None, None]:
    """Loads documents in the given folder"""
    docs_count = 0
    documents = []

    for file in folder.glob("**/*.py"):
        if file.name == "__init__.py":
            continue

        if file.is_file():
            documents.append(Blob(data=file.read_bytes(), path=file.name))
            docs_count += 1

            if docs_count % docs_per_iter == 0:
                yield documents
                documents = []

    if len(documents) > 0:
        yield documents

    if docs_count == 0:
        raise ValueError("No documents found in folder")

    logger.info(f"Loaded {docs_count} documents from {folder}")


def read_documents(documents_generator: Generator) -> Generator:
    """Embeds documents in the given folder"""
    parser = TextParser()

    for documents_blob in documents_generator:
        doc_ids = []
        documents = []
        for document in documents_blob:
            parsed_docs = parser.parse(document)
            doc_id = str(uuid.uuid4())
            for doc in parsed_docs:
                doc.metadata["doc_id"] = doc_id
            documents.extend(parsed_docs)
            doc_ids.append(doc_id)
        yield documents


def embed_documents(
    documents_generator: Generator,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    persist_directory=root / "data" / ".chroma/",
):
    # Create vectorstore for documents
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    vectorstore = Chroma(
        collection_name="documents",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=str(persist_directory),
    )

    for documents in documents_generator:
        split_docs = text_splitter.split_documents(documents)

        if len(split_docs) == 0:
            raise ValueError("Documents couldn't be read")

        logger.info(f"Split into {len(split_docs)} documents")
        logger.info("Embedding documents...")

        vectorstore.add_documents(split_docs)

    vectorstore.persist()
    logger.info(f"Persisted vectorstore to {persist_directory}")
    vectorstore = None


def build_vectorstore(
    persist_directory: Path,
    documents_folder: Path,
    chunk_size=2000,
    docs_per_iter=25,
):
    logger.info(f"Building vectorstore with chunk size: {chunk_size}")

    delete_existing_vectorstore(persist_directory)

    documents_blob = load_documents_from_folder(documents_folder, docs_per_iter)

    documents = read_documents(documents_blob)

    start_time = time.time()
    embed_documents(
        documents,
        chunk_size=chunk_size,
        persist_directory=persist_directory,
    )
    logger.info(f"embed_documents took {time.time() - start_time:.2f} seconds")

    logger.info("Successfully built vectorstore")


if __name__ == "__main__":
    logger.info(f"Root: {root}")
    persist_directory = root / "data" / ".chroma/"
    documents_folder = root / "data" / "documents"
    build_vectorstore(persist_directory, documents_folder)
