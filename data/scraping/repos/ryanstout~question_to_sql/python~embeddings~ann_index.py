import os
from threading import Lock
from typing import Union

import numpy as np

from python.embeddings.ann_faiss import AnnFaiss
from python.embeddings.embedding import generate_embedding
from python.embeddings.embedding_link_index import EmbeddingLinkIndex
from python.embeddings.openai_embedder import OpenAIEmbedder
from python.utils.db import application_database_connection
from python.utils.logging import log

from prisma import Prisma

db: Prisma = application_database_connection()

# ann = approximate nearest neighbor
class AnnIndex:
    def __init__(self, path: str):
        self.index_offset = 0
        self.embeddings = []
        self.lock = Lock()

        self.path = path
        self.embedding_link_index = EmbeddingLinkIndex(path)

    def add(
        self,
        content: str,
        table_id: Union[int, None],
        column_id: Union[int, None],
        value: Union[str, None],
    ):
        log.debug("generating embedding for ", content=content, table=table_id, column=column_id, value=value)
        embedding = generate_embedding(content, embedder=OpenAIEmbedder)

        # Needs the mutex to prevent parallel columns from adding to it at the
        # same time. The index_offset is important for the vector indexing.
        self.lock.acquire()

        previous_offset = self.index_offset
        self.embedding_link_index.add(previous_offset, table_id, column_id, value)
        self.embeddings.append(embedding)
        self.index_offset += 1

        self.lock.release()

    def save(self):
        embed_size = len(self.embeddings)
        if embed_size == 0:
            return

        # TODO what is this doing?
        data = np.stack(self.embeddings, axis=0)

        # Make output folder if it doesn't exist
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        self.embedding_link_index.save()

        log.info(
            "build faiss index",
            dtype=data.dtype,
            size=data.shape,
            embed_size=embed_size,
            path=self.path,
        )
        AnnFaiss().build_and_save(data, self.path)
