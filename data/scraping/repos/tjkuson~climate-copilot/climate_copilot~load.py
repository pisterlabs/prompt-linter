"""Load the resources into Pinecone."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

if TYPE_CHECKING:
    from langchain.schema import Document

    from climate_copilot.utils import PineconeEnvironment


def ingest_resources(directory: Path) -> list[Document]:
    """Ingest the resources from the given directory."""
    loader = PyPDFDirectoryLoader(str(directory))
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n", "\r\n"],
    )
    return loader.load_and_split(text_splitter)


def load_resources(pinecone_env: PineconeEnvironment) -> None:
    """Load the resources into Pinecone."""
    print("Connecting to Pinecone...")
    pinecone.init(api_key=pinecone_env.api_key, environment=pinecone_env.environment)
    embeddings = OpenAIEmbeddings()  # type: ignore[call-arg]

    print("Loading resources...")
    resources_dir = Path(__file__).parent / "resources"
    docs = ingest_resources(resources_dir)
    Pinecone.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=pinecone_env.index_name,
    )

    print("Loaded resources into Pinecone.")
