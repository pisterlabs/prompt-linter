import sys

import pandas as pd
from llama_index import Document, set_global_service_context, StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.indices.base import BaseIndex
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore

from config import (
    API_KEY,
    DEPLOYMENT_NAME,
    MODEL_NAME,
    API_BASE,
    API_VERSION,
    EMBEDDING_MODEL,
    EMBEDDING_DEPLOYMENT,
)

class LlamaQueryEngine:

    def __init__(
            self,
            api_key=API_KEY,
            deployment_name=DEPLOYMENT_NAME,
            model_name=MODEL_NAME,
            api_base=API_BASE,
            api_version=API_VERSION,
            embedding_model=EMBEDDING_MODEL,
            embedding_deployment=EMBEDDING_DEPLOYMENT,
    ):
        import openai
        import logging
        import os

        from langchain.embeddings import OpenAIEmbeddings
        from llama_index.llms import AzureOpenAI
        from llama_index import LangchainEmbedding
        from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        openai.api_type = "azure"
        openai.api_base = api_base
        openai.api_version = api_version
        os.environ["OPENAI_API_KEY"] = api_key
        openai.api_key = os.getenv("OPENAI_API_KEY")

        llm = AzureOpenAI(
            deployment_name=deployment_name,
            model=model_name,
            temperature=0,
            engine="gpt35",
            max_tokens=2048
        )

        embedding_llm = LangchainEmbedding(
            OpenAIEmbeddings(
                model=embedding_model,
                deployment=embedding_deployment,
                openai_api_key=openai.api_key,
                openai_api_base=openai.api_base,
                openai_api_type=openai.api_type,
                openai_api_version=openai.api_version,
            ),
            embed_batch_size=1,
        )

        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embedding_llm,
        )

        set_global_service_context(service_context)
        # index = VectorStoreIndex.from_documents(documents)
        # self.index = index
        # self.query_engine = index.as_query_engine()
        self.index = None
        self.query_engine = None

    def load_doc_from_csv(self, csv_path, text_column="decoded_readme", max_docs=20, is_persist=False, has_persist=False, persist_dir="app/data/persist"):
        if has_persist:
            self.retrieve_index(persist_dir)
            return
        df = pd.read_csv(csv_path)
        text_list = df[text_column].tolist()
        text_list = text_list[:max_docs]
        documents = [Document(text=t) for t in text_list]
        index = VectorStoreIndex.from_documents(documents)
        self.index = index

        from llama_index.indices.postprocessor import SimilarityPostprocessor
        from llama_index.query_engine import RetrieverQueryEngine
        from llama_index.indices.vector_store import VectorIndexRetriever
        from llama_index import get_response_synthesizer

        # configure retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=2,
        )

        # configure response synthesizer
        response_synthesizer = get_response_synthesizer()

        # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.7)
            ]
        )
        self.query_engine = query_engine
        # self.query_engine = index.as_query_engine()
        if is_persist:
            self.persist_index(persist_dir)

    def retrieve_index(self, persist_dir):
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir=persist_dir),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir=persist_dir),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir=persist_dir),
        )
        self.index = load_index_from_storage(storage_context)
        self.query_engine = self.index.as_query_engine()

    def persist_index(self, persist_dir):
        self.index.storage_context.persist(persist_dir=persist_dir)

    def query(self, query_text):
        if not self.query_engine:
            raise Exception("No query engine loaded")
        return self.query_engine.query(query_text)

    def get_index(self):
        return self.index
