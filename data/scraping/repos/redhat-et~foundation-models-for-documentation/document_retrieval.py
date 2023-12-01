# Import necessary libraries
import numpy as np
import json
#from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
import os

class DocumentRetrieval:
    def __init__(self, config_path):
        
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
        
        # Load the embedding model
        self.embeddings = SentenceTransformerEmbeddings(model_name=self.config["embedding_model_name"])
        
        # Load the vector store if it exists
        if os.path.isdir(os.path.join(self.config["vdb_path"],
                                      "collection",
                                      self.config["vdb_collection_name"])):
            self.vector_store = self.load_vector_store()
        # Otherwise, create it
        else:
            self.vector_store = self.create_vector_store()

    def load_documents(self, dataset_path):
        # Load your dataset from the given path and return a list of documents
        print("Loading documents")
        loader = DirectoryLoader(dataset_path,
                                 glob="**/*.md",
                                 loader_cls=TextLoader)
        documents = loader.load()
        return documents
    
    def load_vector_store(self):
        print("Loading vector store")
        client = QdrantClient(path=self.config["vdb_path"])
        qdrant_index = Qdrant(client,
                        self.config["vdb_collection_name"],
                        self.embeddings)
        return qdrant_index
        

    def create_vector_store(self):
        # Load dataset
        self.documents = self.load_documents(self.config["dataset_path"])
        # Encode all documents and store their embeddings in a dictionary
        text_splitter = MarkdownTextSplitter(chunk_size=1000,
                                             chunk_overlap=0)

        texts = text_splitter.split_documents(self.documents)

        qdrant_index = Qdrant.from_documents(
            texts,
            self.embeddings,
            path=self.config["vdb_path"],  # Local mode with in-memory storage only
            collection_name="rosa_documents",
            force_recreate=False
        )
        return qdrant_index

    def search_candidates(self, user_query, top_k=5):
        # Encode the user query
        candidate_documents = self.vector_store.similarity_search(user_query)
        sources = [i.metadata['source'] for i in candidate_documents]
        return candidate_documents, sources
