import logging
from typing import List

import pinecone
from icecream import ic
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Pinecone

from config.constants import INDEX_NAME


def index_list() -> List[str]:
    return pinecone.list_indexes()


def create_index(index_name: str = INDEX_NAME, dimension: int = 1536, metric: str = 'euclidean'):
    # check before creating
    if index_name not in index_list():
        # index not existed. Create a new index
        pinecone.create_index(
            name=index_name, dimension=dimension, metric=metric,
        )
        ic(f'created a new index {index_name}')
    else:
        logging.warning(f'{index_name} index existed. skip creating.')


def insert(data: List[Document], embeddings: OpenAIEmbeddings, index=INDEX_NAME) -> Pinecone:
    return Pinecone.from_documents(data, embedding=embeddings, index_name=index)


def need_text_embedding():
    need_text_embedding = False
    index = pinecone.Index(INDEX_NAME)
    index_stats_response = index.describe_index_stats()
    # Example of index_stats_reponse
    # index_stats_response is {'dimension': 1536,
    # 'index_fullness': 0.0,
    # 'namespaces': {'': {'vector_count': 532}},
    # 'total_vector_count': 532}
    vector_count = index_stats_response.total_vector_count
    print(f'total_vector_count is {vector_count}')

    if vector_count == 0:
        need_text_embedding = True
    return need_text_embedding
