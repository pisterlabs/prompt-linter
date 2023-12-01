import sqlite3
import faiss
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from repository import Repository
import typing

class DocumentIndex:
    INDEX_WINDOW = 10
    TEXT_EMBEDDING_ADA_002_DIMENSION = 1536
    def __init__(self, channel_id):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.index: Optional[faiss.IndexFlatL2] = None
        self.channel_id = channel_id
        self.repository = Repository(channel_id)

    def add_message(self, message, unix_timestamp: int):
        if self.index is None:
            self.load_or_create_index()
        # This fills mypy with joy
        assert self.index is not None

        document_embedding = self.embeddings.embed_documents([message])[0]
        serialized_embedding = ",".join(map(str, document_embedding))

        self.repository.save_long_term_memory(message, unix_timestamp, serialized_embedding)
        self.index.add(np.array([document_embedding]).astype('float32'))

        self.repository.save_long_term_memory_index(self.index)

    def load_or_create_index(self):
        possible_index = self.repository.load_long_term_memory_index()
        if possible_index:
            self.index = possible_index
        else:
            self.index = faiss.IndexFlatL2(self.TEXT_EMBEDDING_ADA_002_DIMENSION)
            self.repository.save_long_term_memory_index(self.index)

    def search_index(self, query, threshold=0.5, token_threshold=500):
        if self.index is None:
            self.load_or_create_index()
        query_embedding = self.embeddings.embed_query(query)
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), self.INDEX_WINDOW)
        results = []
        total_token_count = 0
        for distance, index in zip(distances[0], indices[0]):
            if 0 <= distance < threshold and index != -1:
                memory_id = int(index) + 1
                memory = self.repository.load_memory(memory_id)
                if memory is not None:
                    total_token_count += memory.text_token_count
                    if total_token_count > token_threshold:
                        break
                    results.append(memory)

        return results

    def rebuild_index(self):
        embeddings = self.repository.load_embeddings()
        if embeddings:
            self.index = faiss.IndexFlatL2(self.TEXT_EMBEDDING_ADA_002_DIMENSION)
            self.index.add(np.array(embeddings).astype('float32'))
            self.repository.save_long_term_memory_index(self.index)



