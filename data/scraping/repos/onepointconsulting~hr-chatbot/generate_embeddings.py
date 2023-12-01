from langchain.schema import Document
from langchain.document_loaders import PyPDFium2Loader
from langchain.vectorstores import FAISS

from typing import TypeVar, List
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

import os
import re

from config import cfg

from log_init import logger

load_dotenv()

VST = TypeVar("VST", bound="VectorStore")


def load_pdfs(path: Path) -> List[Document]:
    """
    Loads the PDFs and extracts a document per page.
    The page details are added to the extracted metadata

    Parameters:
    path (Path): The path where the PDFs are saved.

    Returns:
    List[Document]: Returns a list of values
    """
    assert path.exists()
    all_pages = []
    for pdf in path.glob("*.pdf"):
        loader = PyPDFium2Loader(str(pdf.absolute()))
        pages: List[Document] = loader.load_and_split()
        for i, p in enumerate(pages):
            file_name = re.sub(r".+[\\/]", "", p.metadata["source"])
            p.metadata["source"] = f"{file_name} page {i + 1}"
        all_pages.extend(pages)
        logger.info(f"Processed {pdf}, all_pages size: {len(all_pages)}")
    log_stats(all_pages)
    return all_pages


def log_stats(documents: List[Document]):
    logger.info(f"Total number of documents {len(documents)}")
    counts = []
    for d in documents:
        counts.append(count_words(d))
    logger.info(f"Tokens Max {np.max(counts)}")
    logger.info(f"Tokens Min {np.min(counts)}")
    logger.info(f"Tokens Min {np.mean(counts)}")


def generate_embeddings(
    documents: List[Document], path: Path, faiss_persist_directory: str
) -> VST:
    """
    Receives a list of documents and generates the embeddings via OpenAI API.

    Parameters:
    documents (List[Document]): The document list with one page per document.
    path (Path): The path where the documents are found.

    Returns:
    VST: Recturs a reference to the vector store.
    """
    try:
        docsearch = FAISS.from_documents(documents, cfg.embeddings)
        docsearch.save_local(faiss_persist_directory)
        logger.info("Vector database persisted")
    except Exception as e:
        logger.error(f"Failed to process {path}: {str(e)}")
        if "docsearch" in vars() or "docsearch" in globals():
            docsearch.persist()
        return docsearch
    return docsearch


def count_words(document: Document) -> int:
    splits = [s for s in re.split("[\s,.]", document.page_content) if len(s) > 0]
    return len(splits)


if __name__ == "__main__":
    doc_location: str = os.environ["DOC_LOCATION"]
    documents = load_pdfs(Path(doc_location))
    assert len(documents) > 0
    logger.info(documents[2].page_content)
    generate_embeddings(documents, doc_location)
