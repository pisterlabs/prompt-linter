# import logging
import sys
import os
import warnings
import re
import math
import hashlib
from typing import List, Optional

# import gradio as gr
import numpy as np
import sqlalchemy
from sqlalchemy.orm import Session
from sqlalchemy import desc

from pydantic import BaseModel, Field, root_validator

from langchain.schema import Document
from langchain.schema import BaseRetriever
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.pgvector import EmbeddingStore, PGVector
from langchain.embeddings.base import Embeddings

sys.path.append("./")
from src.models import LlamaModelHandler
from src.prompts.memory import IMPORTANCE_RATING_PROMPT
from langchain.text_splitter import TextSplitter

from src.util import get_secrets, get_default_text_splitter, get_epoch_time

# suppress warnings for demo
warnings.filterwarnings("ignore")
os.environ["PYDEVD_INTERRUPT_THREAD_TIMEOUT"] = "60"
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT "] = "60"
os.environ["MYLANGCHAIN_SAVE_CHAT_HISTORY"] = "0"


class PGMemoryStoreBase:
    """set up a long term memory handling for an agent executor to use as document tools"""

    connection_string: str = None
    memory_collection: str = None
    memory_collection_meta: str = None
    doc_embedding: Embeddings = None
    pg_vector_store: PGVector = None

    def __init__(self, embedding, memory_collection="long_term"):
        _postgres_host = get_secrets("postgres_host")
        if _postgres_host is not None:
            os.environ["PGVECTOR_HOST"] = _postgres_host.split(":")[0]
            os.environ["PGVECTOR_PORT"] = _postgres_host.split(":")[1]
        _postgres_db = get_secrets("postgres_db")
        if _postgres_db is not None:
            os.environ["PGVECTOR_DATABASE"] = _postgres_db
        _postgres_user = get_secrets("postgres_user")
        if _postgres_user is not None:
            os.environ["PGVECTOR_USER"] = _postgres_user
        _postgres_pass = get_secrets("postgres_pass")
        if _postgres_pass is not None:
            os.environ["PGVECTOR_PASSWORD"] = _postgres_pass
        self.connection_string = PGVector.connection_string_from_db_params(
            driver=os.environ.get("PGVECTOR_DRIVER", default="psycopg2"),
            host=os.environ.get("PGVECTOR_HOST", default="localhost"),
            port=int(os.environ.get("PGVECTOR_PORT", default="5432")),
            database=os.environ.get("PGVECTOR_DATABASE", default="postgres"),
            user=os.environ.get("PGVECTOR_USER", default="postgres"),
            password=os.environ.get("PGVECTOR_PASSWORD", default="postgres"),
        )
        self.memory_collection = memory_collection
        self.memory_collection_meta = {
            "description": "all of the long term memory stored as part of langchain agent runs."
        }
        self.doc_embedding = embedding
        self.pg_vector_store = PGVector(
            connection_string=self.connection_string,
            embedding_function=self.doc_embedding,
            collection_name=self.memory_collection,
            collection_metadata=self.memory_collection_meta,
        )


class PGMemoryStoreSetter(PGMemoryStoreBase):
    def add_memory(
        self,
        llm,
        text,
        prompt="",
        thought="",
        with_importance=True,
        type="memory",
        retrieval_eligible="True",
    ):
        """Adds text to memory store. Assigns store time and initializes access time. Optionally, use llm to generate importance score. Assigns custom_id which is a hash of the store time and text.

        Args:
            text (str): text to add to memory
            llm (LLM): llm to use to generate importance score
            with_importance (bool, optional): if True, generate importance score.
            type (str, optional): species type of memory from ["memory", "reflection", "identity", "plan"]
            retrieval_eligible (str, optional): if not "True", will not be retrieved unless type (e.g. "identity") specified in retrieval.

        """
        # build texts object for memory
        memory_texts = [text]
        store_time = str(get_epoch_time())
        access_time = store_time

        if with_importance:
            importance_rating_text = llm(
                IMPORTANCE_RATING_PROMPT.replace("{memory}", text)
            )
            re_result = re.search(
                r"\d+(?=\D|$)", importance_rating_text
            )  # search for the first digits until you find a non-digit or end of string
            if re_result:
                # if match is found
                importance_rating = int(re_result.group()) / 10
            else:
                # if no match is found
                importance_rating = 0.2
        else:
            importance_rating = 0.2

        # generate hash for id
        id_text = f"{store_time}_{text}"
        id_hash = hashlib.md5(id_text.strip().encode()).hexdigest()

        # build metadata
        json_metadata = [
            {
                "custom_id": id_hash,
                "store_time": store_time,
                "access_time": access_time,
                "prompt": prompt,
                "thought": thought,
                "type": type,
                "importance": importance_rating,
                "retrieval_eligible": retrieval_eligible,
            }
        ]

        # method 1: build custom text
        self.pg_vector_store.add_texts(
            texts=memory_texts, metadatas=json_metadata, ids=[id_hash]
        )

        # method 2: use splitter and from documents
        # splitter = get_default_text_splitter("character")
        # new_memory_doc = splitter.create_documents(texts = memory_texts, metadatas=json_metadata)
        # new_memory_doc_split = splitter.split_documents([new_memory_doc])
        # db = PGVector.from_documents(
        #     embedding=self.doc_embedding,
        #     documents=new_memory_doc_split,
        #     collection_name=self.memory_collection,
        #     connection_string=self.connection_string,
        # )


class PGMemoryStoreRetriever(PGMemoryStoreBase):
    def update_memory(self, custom_id):
        """Retrieves memory based on its custom_id field.

        Args:
            custom_id (str): custom id

        Returns:
            _type_: _description_
        """
        # retrieves memory and updates access time
        new_access_time = str(get_epoch_time())

        engine = sqlalchemy.create_engine(self.connection_string)
        conn = engine.connect()
        session = Session(conn)

        filter_by = EmbeddingStore.cmetadata["custom_id"].astext == custom_id

        memory_record = session.query(EmbeddingStore).filter(filter_by)
        # memory_doc = memory_record[0].document
        memory_record_cmetadata = memory_record[0].cmetadata
        memory_record_cmetadata["access_time"] = new_access_time
        memory_record[0].cmetadata = memory_record_cmetadata
        session.commit()

        # return memory_record[0]

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        mem_to_search = kwargs["mem_to_search"] if "mem_to_search" in kwargs else 30
        mem_to_return = kwargs["mem_to_return"] if "mem_to_return" in kwargs else 3
        relevance_wt = kwargs["relevance_wt"] if "relevance_wt" in kwargs else 1
        importance_wt = kwargs["importance_wt"] if "importance_wt" in kwargs else 1
        recency_wt = kwargs["recency_wt"] if "recency_wt" in kwargs else 1
        mem_type = kwargs["mem_type"] if "mem_type" in kwargs else "all"
        update_access_time = (
            kwargs["update_access_time"] if "update_access_time" in kwargs else True
        )
        return self.retrieve_memory_list(
            query=query,
            mem_to_search=mem_to_search,
            mem_to_return=mem_to_return,
            relevance_wt=relevance_wt,
            importance_wt=importance_wt,
            recency_wt=recency_wt,
            mem_type=mem_type,
            update_access_time=update_access_time,
        )

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("PGMemoryStoreRetriever does not support async")

    def retrieve_memory_list(
        self,
        query,
        mem_to_search=30,
        mem_to_return=3,
        relevance_wt=1,
        importance_wt=1,
        recency_wt=1,
        mem_type="all",
        update_access_time=True,
    ):
        """Retrieves a list of memory objects, based on priority score which is a function of relevance, importance and recency, out of the k most relevant memories, where k=mem_to_search.

        Args:
            query (_type_): _description_
            mem_to_search (int, optional): k memories to rank and select from. Defaults to 30.
            mem_to_return (int, optional): number of memory objects to return. Defaults to 3.
            relevance_wt (int, optional): relevance weight. Defaults to 1.
            importance_wt (int, optional): importance weight. Defaults to 1.
            recency_wt (int, optional): recency weight. Defaults to 1.
            mem_type (str, optional): type of memory to return. Defaults to "all"
            update_access_time (boolean, optional): if true, update the access_time. Defaults to true.

        Returns:
            list: memory objects
        """

        # add filter = {"type": "reflection"}
        if mem_type != "all":
            result = self.pg_vector_store.similarity_search_with_score(
                query, k=mem_to_search, filter={"type": mem_type}
            )
        else:
            result = self.pg_vector_store.similarity_search_with_score(
                query, k=mem_to_search, filter={"retrieval_eligible": "True"}
            )

        custom_id = []
        memory_content = []
        relevance_score = []
        importance_score = []
        recency_score = []
        n_result = len(result)

        for rank in range(n_result):
            memory_content.append(result[rank][0].page_content)
            custom_id.append(result[rank][0].metadata["custom_id"])

            # relevance
            distance = result[rank][1]
            relevance_score.append(
                1 - ((1 / (1 + math.exp((-distance + 1) * 3))))
            )  # convert distance to relevance score

            # importance
            importance_score.append(result[rank][0].metadata["importance"])

            # recency score
            half_life = 604800  # 1 week in seconds
            elapsed_since_last_access_s = get_epoch_time() - float(
                result[rank][0].metadata["access_time"]
            )
            recency_score.append((0.5) ** (elapsed_since_last_access_s / half_life))

        priority_product_score = np.array(
            [
                (a**relevance_wt) * (b**importance_wt) * (c**recency_wt)
                for a, b, c in zip(relevance_score, importance_score, recency_score)
            ]
        )
        priority_indices = np.argsort(priority_product_score)
        max_priority_indices = priority_indices[-mem_to_return:]
        max_priority_indices = np.flip(max_priority_indices)
        max_priority_memory_ids = [custom_id[i] for i in max_priority_indices]
        max_priority_memories = [result[i][0] for i in max_priority_indices]

        if update_access_time:
            for i in max_priority_memory_ids:
                self.update_memory(i)

        return max_priority_memories


if __name__ == "__main__":
    # model_name = "llama-13b"
    # lora_name = "alpaca-gpt4-lora-13b-3ep"
    model_name = "llama-7b"
    lora_name = "alpaca-lora-7b"
    testAgent = LlamaModelHandler()
    eb = testAgent.get_hf_embedding()

    pipeline, model, tokenizer = testAgent.load_llama_llm(
        model_name=model_name, lora_name=lora_name, max_new_tokens=200
    )

    memory_store_setter = PGMemoryStoreSetter(embedding=eb)
    memory_store_retriever = PGMemoryStoreRetriever(embedding=eb)

    # updates the access time
    # memory_store.update_memory("fd91b4902786c06621a2a3e03dd3ad1e")

    # Add a memory. If with_importance is True, llm call to generate importance
    memory_store_setter.add_memory(
        "I am reflecting on my life. I remember this",
        llm=pipeline,
        with_importance=True,
        type="memory",
        retrieval_eligible="True",
    )

    # Retrieve list of memory objects prioritized by relevance*importance*recency
    # memory_list = memory_store.retrieve_memory_list(
    #     query="penguins",
    #     mem_to_search=100,
    #     mem_to_return=5,
    #     relevance_wt=1,
    #     importance_wt=1,
    #     recency_wt=1,
    # )
    memory_kwargs = {
        "mem_to_search": 30,
        "mem_to_return": 1,
        "recency_wt": 1,
        "importance_wt": 0,
        "relevance_wt": 0,
        "mem_type": "identity",
    }
    memory_list = memory_store_retriever.get_relevant_documents(
        "",
        **memory_kwargs,
    )
    print(memory_list)

    print("done")
