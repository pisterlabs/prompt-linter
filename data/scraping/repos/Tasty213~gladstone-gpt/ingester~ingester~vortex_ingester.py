from hashlib import sha256
import json
from pathlib import Path
from typing import List
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import langchain.docstore.document as docstore
import logging as logger
from parsers.json import JsonParser
from tqdm import tqdm as progress_bar

COLLECTION_NAME = "neonshield-2023-05"
PERSIST_DIRECTORY = "./temp_data/chroma"


class VortexIngester:
    def __init__(self, content_folder: str):
        self.content_folder = content_folder

    def ingest(
        self,
        chunk_size=250,
        model_name="gpt-3.5-turbo",
        chunk_overlap=25,
        split_document_text=True,
    ) -> None:
        files_to_ingest = list(Path(self.content_folder).glob("*.json"))
        vortex_json_parser = JsonParser(
            chunk_size=chunk_size,
            model_name=model_name,
            chunk_overlap=chunk_overlap,
            split_document_text=split_document_text,
        )

        chunks: List[docstore.Document] = []
        for document in progress_bar(files_to_ingest, total=len(files_to_ingest)):
            try:
                new_chunks = vortex_json_parser.text_to_docs(
                    document, self.content_folder
                )
                logger.debug(f"Extracted {len(new_chunks)} chunks from {document}")
                chunks.extend(new_chunks)
            except Exception as e:
                logger.error(f"failed to ingest {document} because {e}")

        embeddings = OpenAIEmbeddings(client=None)
        logger.info("Loaded embeddings")
        vector_store = Chroma.from_documents(
            chunks,
            embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIRECTORY,
            ids=self.get_sha_of_chunks(chunks),
        )

        logger.info("Created Chroma vector store")
        vector_store.persist()
        logger.info("Persisted Chroma vector store")

    @staticmethod
    def get_sha_of_chunks(chunks: List[docstore.Document]):
        ids = set()

        for chunk in chunks:
            sha = sha256(chunk.page_content.encode("utf-8")).hexdigest()
            if sha in ids:
                raise ValueError(
                    f"Found ID {sha} twice current document is {json.dumps(chunk.metadata)} content is {chunk.page_content}"  # pylint: disable=line-too-long
                )
            ids.add(sha)

        return list(ids)
