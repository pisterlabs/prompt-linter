from astrapy.db import AstraDB
from langchain.embeddings import CohereEmbeddings, OpenAIEmbeddings
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import AstraDB as LCAstraDB

import gradio as gr

from loguru import logger
import os
import time

model_to_dimension = {
    "cohere_english_3": 1024,
    "cohere_english_light_3": 384,
    "cohere_multilingual_3": 1024,
    "cohere_multilingual_light_3": 384,
    "openai": 1536,
}

# TODO: this class isn't really a proper astra wrapper, blended astra + embedding model concepts
#       should separate these out
class Astra:
    def __init__(self, state) -> None:
        self.token = os.environ["ASTRA_TOKEN"]
        self.astra_endpoint = os.environ["ASTRA_API_ENDPOINT"]
        self.collection_name = os.environ["ASTRA_COLLECTION"]
        self.db = AstraDB(token=self.token, api_endpoint=self.astra_endpoint)
        self.vectorstore = None
        self.current_collection_model = None
        self.state = state
    
    def _create_vectorstore(self, embed_model):
        # user can change embedding model in UI, so init this at runtime of task
        return LCAstraDB(
            embedding=embed_model,
            collection_name=self.collection_name,
            token=self.token,
            api_endpoint=self.astra_endpoint,
        )

    def _create_embedding_model(self, embedding_model: str) -> Embeddings:

        if type(embedding_model) is list:
            gr.Info("Choose embedding model on models tab.")
            return None
        
        print(embedding_model)
        # TODO: collection creation should probably not be buried in this method
        collections = self.db.get_collections()
        if self.collection_name not in collections["status"]["collections"]:
            self.db.create_collection(
                self.collection_name, 
                dimension=model_to_dimension[embedding_model], 
                metric="cosine")
            self.current_collection_model = embedding_model

        if embedding_model != self.current_collection_model:
            self.db.delete_collection(collection_name=self.collection_name)
            self.db.create_collection(
                self.collection_name, 
                dimension=model_to_dimension[embedding_model], 
                metric="cosine")

        if embedding_model == "openai":
            try:     
                return OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
            except Exception as e:
                logger.warning(f"{e}")
                raise ValueError("Invalid OpenAI API key.")

        elif embedding_model == "cohere_english_3":
            try:
                return CohereEmbeddings(cohere_api_key=os.environ["COHERE_API_KEY"], model="embed-english-v3.0")
            except Exception as e:
                logger.warning(f"{e}")
                raise ValueError("Invalid Cohere API key.")
            
        elif embedding_model == "cohere_english_light_3":
            try:
                return CohereEmbeddings(cohere_api_key=os.environ["COHERE_API_KEY"], model="embed-english-light-v3.0")
            except Exception as e:
                logger.warning(f"{e}")
                raise ValueError("Invalid Cohere API key.")
            
        elif embedding_model == "cohere_multilingual_3":
            try:
                return CohereEmbeddings(cohere_api_key=os.environ["COHERE_API_KEY"], model="embed-multilingual-v3.0")
            except Exception as e:
                logger.warning(f"{e}")
                raise ValueError("Invalid Cohere API key.")
            
        elif embedding_model == "cohere_multilingual_light_3":
            try:
                return CohereEmbeddings(cohere_api_key=["COHERE_API_KEY"], model="embed-multilingual-light-v3.0")
            except Exception as e:
                logger.warning(f"{e}")
                raise ValueError("Invalid Cohere API key.")
            
        else:
            raise ValueError(f"Invalid embedding model. Set on models tab.")

    def get_relevant_documents_reranker(
            self, 
            query: str, 
            embedding_model: str, 
            reranker: str,
            n=8):
        
        embed_model = self._create_embedding_model(embedding_model)
        self.vectorstore = self._create_vectorstore(embed_model)
        retriever = self.vectorstore.as_retriever()

        docs = []
        if reranker == "cohere":
            compressor = CohereRerank(
                cohere_api_key=os.environ["COHERE_API_KEY"], 
                user_agent="rag-playground",
                top_n=n,
            )
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )
            docs = compression_retriever.get_relevant_documents(query)

        elif reranker == "mmr":
            example_selector = MaxMarginalRelevanceExampleSelector(
                fetch_k=30,
                k=n,
                vectorstore=self.vectorstore,
            )
            docs = retriever.get_relevant_documents(query)
            logger.info("\n\n".join([str(_) for _ in docs]))
            docs = example_selector.select_examples(docs)
            docs = docs[:n]

        elif reranker == "none":
            docs = ["No reranking"]

        return "\n\n".join([str(_) for _ in docs])
        

    def get_relevant_documents(self, query, embedding_model: str, n=8):
        embed_model = self._create_embedding_model(embedding_model)
        self.vectorstore = self._create_vectorstore(embed_model)

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": n})
        docs = retriever.get_relevant_documents(query)

        return "\n\n".join([str(_) for _ in docs])

    def get_vectorstore(self):
        return self.vectorstore

    def store_chunks(
        self, 
        embedding_model: str,
    ) -> None:             
      
        gr.Info("Adding chunks to Astra DB.")

        embed_model = self._create_embedding_model(embedding_model)

        if embed_model is None:
            gr.Info("Choose embedding model on models tab.")
            return None

        self.vectorstore = self._create_vectorstore(embed_model)

        logger.info("Adding document to Astra DB.")

        chunks = self.state.get_chunks()
        start = time.time()
        self.vectorstore.add_texts(texts=chunks)
        end = time.time() 

        gr.Info(f"Stored {len(chunks)} embeddings. Avg time per embedding {(end-start)/len(chunks)} seconds.")
        logger.info(f"Added {len(chunks)} embeddings to collection: {self.collection_name}.")
        logger.info(f"Average time per embedding {(end-start)/len(chunks)} seconds.")

    def delete_collection(self) -> None:
        self.db.delete_collection(collection_name=self.collection_name) 

        gr.Info("Deleted embeddings collection.")
        logger.info(f"Deleted collection: {self.collection_name}.")  