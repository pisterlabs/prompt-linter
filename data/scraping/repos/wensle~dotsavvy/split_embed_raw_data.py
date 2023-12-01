import json
import pickle
from pathlib import Path
from typing import Generator
from uuid import uuid4

from langchain.document_loaders import PyPDFLoader

from config import ICT_RESEARCH_METHODS_BASE_DIR, PDFS_BASE_DIR
from dotsavvy.datastore.pinecone.types import METADATA
from dotsavvy.services.chunk import create_recursive_tiktoken_splitter
from dotsavvy.services.embedding import embed_documents

_INPUT_DIR: Path = PDFS_BASE_DIR / "raw_data"
_OUTPUT_FILE_PATH: Path = PDFS_BASE_DIR / "processed_data" / "embedding_tuples.pkl"


def ingest_file(filepath: str | Path) -> tuple[list[str], list[METADATA]]:
    """A helper function to load the input file containing processed data."""
    with open(filepath, "r") as f:
        json_obj = json.load(f)
    return json_obj["texts"], json_obj["metadatas"]


def process_dir_generator(
    input_dir: Path,
) -> Generator[tuple[str, METADATA], None, None]:
    for raw_data_file in input_dir.glob("*.pdf"):
        yield str(raw_data_file)


def main(
    input_dir: Path | None = None,
    output_file_path: str | Path | None = None,
) -> None:
    input_dir = input_dir or _INPUT_DIR
    output_file_path = output_file_path or _OUTPUT_FILE_PATH

    text_splitter = create_recursive_tiktoken_splitter()
    chunks = []
    metadatas = []
    uuids = []
    for raw_data_file in process_dir_generator(input_dir):
        loader = PyPDFLoader(raw_data_file)
        print(raw_data_file)
        for document in loader.lazy_load():
            for chunk_id, chunk in enumerate(
                text_splitter.split_text(document.page_content)
            ):
                chunks.append(chunk)
                document.metadata["chunk_id"] = chunk_id
                document.metadata["text"] = chunk
                metadatas.append(document.metadata)
                uuids.append(str(uuid4()))

    embeddings = embed_documents(chunks)
    embedding_tuples = list(zip(uuids, embeddings, metadatas))

    with open(_OUTPUT_FILE_PATH, "wb") as pkl:
        pickle.dump(embedding_tuples, pkl)


if __name__ == "__main__":
    main()
