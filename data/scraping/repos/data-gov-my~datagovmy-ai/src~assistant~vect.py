"""Helper interface class for vectorstores."""

import weaviate
import pandas as pd
from typing import Protocol, List

from langchain.vectorstores import VectorStore
from langchain.embeddings import OpenAIEmbeddings

from langchain.vectorstores.weaviate import Weaviate

from config import *


class BaseVectorDB(Protocol):
    vectorstore: VectorStore

    def setup(self):
        """Initialize vectorstore for the first time"""

    def update(self, new_data: pd.DataFrame) -> List[int]:
        """Add or update records"""

    def remove(self, ids_to_remove: List) -> None:
        """Remove records from vectorstore"""


class WeaviateVectorDB(BaseVectorDB):
    meta_fields: List[str]  # unique to weaviate
    client: weaviate.Client

    def __init__(self, meta_fields: List[str], instance_url: str, index: str):
        self.meta_fields = meta_fields
        oai_embeddings = OpenAIEmbeddings()

        # connect to existing weaviate instance
        self.client = weaviate.Client(url=instance_url)
        self.vectorstore = Weaviate(
            self.client,
            index,
            "text",  # constant
            embedding=oai_embeddings,
            attributes=self.meta_fields,
            by_text=False,  # force vector search
        )

    def update(self, new_data: pd.DataFrame) -> List[int]:
        """Add or update records to vectorstore"""
        new_texts = new_data.content_embed.tolist()
        new_metadatas = new_data[self.meta_fields].to_dict(orient="records")
        new_uuids = new_data.uuid.tolist()

        new_ids = self.vectorstore.add_texts(new_texts, new_metadatas, uuids=new_uuids)
        return new_ids

    def remove(self, ids_to_remove: List) -> None:
        for uuid_remove in ids_to_remove:
            try:
                self.client.data_object.delete(
                    uuid=uuid_remove,
                )
            except Exception as e:
                print(e)
                continue
