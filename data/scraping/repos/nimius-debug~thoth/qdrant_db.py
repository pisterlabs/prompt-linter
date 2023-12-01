
import streamlit as st
import logging
import qdrant_client
from langchain.vectorstores import Qdrant
from qdrant_client.http import models
from langchain.embeddings import OpenAIEmbeddings

class QdrantSingleton:
    _instance = None
    
    def __new__(cls):
        logging.info("Creating a new instance...")
        if cls._instance is None:
                cls._instance = super(QdrantSingleton, cls).__new__(cls)
                cls._instance.initialize_qdrant_client()
        print("Returning the instance...", cls._instance)
        return cls._instance

    def initialize_qdrant_client(self):
        logging.warning("Initializing Qdrant client...")
        QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
        QDRANT_HOST = st.secrets["QDRANT_HOST"]

        self.client = qdrant_client.QdrantClient(
            api_key=QDRANT_API_KEY,
            url=QDRANT_HOST
        )
        existing_collections = self.client.get_collections()  # Assuming this returns a list of existing collections
        print("Existing collections: ", existing_collections)
        username = st.session_state['username']
        
        if username not in existing_collections:
            print("Creating a new collection...")
            self.client.recreate_collection(
                collection_name=username,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
            )
        
    # def collection_exists(self, collection_name):
    #     existing_collections = self.client.get_collections()
    #     return collection_name in existing_collections
    
    def get_vector_store(self, username):
        embeddings = OpenAIEmbeddings(openai_api_key= st.secrets["OPENAI_API_KEY"])
        self.vector_store = Qdrant(
            client=self.client, 
            collection_name=username, 
            embeddings=embeddings
        )
        return self.vector_store
    
    def delete_points_associated_with_file(self, filename, collection_name):
        try:
            # Create a filter for metadata matching the filename
            my_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.filename",
                        match=models.MatchValue(value=filename)
                    ),
                ]
            )

            # Delete vectors with the specified filter
            response = self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(filter=my_filter),
                wait=True  # You may optionally set wait to True for synchronous operation
            )

            print(response)
            
            if response:  # Replace with actual condition based on what Qdrant client returns
                print(f"Deleted all vectors associated with filename: {filename}")
            else:
                print(f"No vectors deleted for filename: {filename}")

        except Exception as e:
            print(f"An error occurred while deleting vectors: {e}")