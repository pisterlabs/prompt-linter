# -*- coding: utf-8 -*-
"""
Created on Tue May 16 22:59:06 2023

@author: marca
"""

import tiktoken
import configparser
import openai
from openai.error import RateLimitError, InvalidRequestError, APIError
import pinecone
from pinecone import PineconeProtocolError
import time
import pandas as pd
from tqdm.auto import tqdm
import sys
import os
from embedder import Embedder
import asyncio
import nest_asyncio

nest_asyncio.apply()

class PineconeMemory:

    def __init__(self, index_name=None, namespace=None):
        
        if not os.path.exists("config.ini"):
            raise FileNotFoundError("The config file was not found.")
        
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.openai_api_key, self.pinecone_api_key, self.pinecone_env, self.index_name, self.namespace = self._get_api_keys("config.ini")
        if index_name:
            self.index_name = index_name
        if namespace:
            self.namespace = namespace
        self.pending_data = None
        pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_env)
        
        if self.index_name not in pinecone.list_indexes():
            dimension = 1536
            metric = "cosine"
            pod_type = "p1"
            
            pinecone.create_index(
                    self.index_name, dimension=dimension, metric=metric, pod_type=pod_type
                    )
            
        self.index = pinecone.Index(index_name=self.index_name)
        openai.api_key = self.openai_api_key
        
        self.embedder = Embedder()

    def __str__(self):
        """Returns a string representation of the PineconeMemory object."""
        return f"Pinecone Memory | Index: {self.index_name}"
        
        
    def _get_api_keys(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        
        openai_api_key = config.get("API_KEYS", "OpenAI_API_KEY")
        pinecone_api_key = config.get("API_KEYS", "Pinecone_API_KEY")
        pinecone_env = config.get("API_KEYS", "Pinecone_ENV")
        index = config.get("API_KEYS", "Pinecone_Index")
        namespace = config.get("API_KEYS", "Pinecone_Namespace")

        return openai_api_key, pinecone_api_key, pinecone_env, index, namespace

    def _count_tokens(self, text):
        tokens = len(self.encoding.encode(text))
        return tokens
    
    def store_single(self, chunk: str):
        """Store a single embedding in Pinecone."""
        
        assert self._count_tokens(chunk) <= 1200, "Text too long, chunk text before passing to .store_single()"
        
        vector = self.embedder.get_embedding(chunk)
        idx = self.index.describe_index_stats()["namespaces"][self.namespace]['vector_count'] + 1

        # Prepare metadata for upsert
        metadata = {"context": chunk}
        vectors_to_upsert = [(idx, vector, metadata)]

        # Attempt to upsert the vector to Pinecone
        while True:
            try:
                upsert_response = self.index.upsert(
                    vectors=vectors_to_upsert, namespace=self.namespace
                )
                break
            except pinecone.core.client.exceptions.ApiException:
                print("Pinecone is a little overwhelmed, trying again in a few seconds...")
                time.sleep(10)


    def store(self, context_chunks: list):
        
        
        if context_chunks:
            batch_size = 80
            vectors_to_upsert = []
            batch_count = 0
            start_id = self.index.describe_index_stats()["namespaces"][self.namespace]['vector_count'] + 1
            data = asyncio.run(self.embedder.create_embeddings(context_chunks, start_id=start_id))
            # Calculate the total number of batches
            total_batches = -(-len(data) // batch_size)
    
            # Create a tqdm progress bar object
            progress_bar = tqdm(total=total_batches, desc="Loading info into Pinecone", position=0)
    
            for index, row in data.iterrows():
                context_chunk = row["chunk"]
    
                vector = row["embeddings"]
    
                pine_index = str(row["id"])
                metadata = {"context": context_chunk}
                vectors_to_upsert.append((pine_index, vector, metadata))
    
                # Upsert when the batch is full or it's the last row
                if len(vectors_to_upsert) == batch_size or index == len(data) - 1:
                    while True:
                        try:
                            upsert_response = self.index.upsert(
                                vectors=vectors_to_upsert, namespace=self.namespace
                            )
    
                            batch_count += 1
                            vectors_to_upsert = []
    
                            # Update the progress bar
                            progress_bar.update(1)
                            sys.stdout.flush()
                            break
    
                        except pinecone.core.client.exceptions.ApiException:
                            print(
                                "Pinecone is a little overwhelmed, trying again in a few seconds..."
                            )
                            time.sleep(10)
    
            # Close the progress bar after completing all upserts
            progress_bar.close()
            
    
        else:
            print("No dataframe to retrieve embeddings")


    def fetch_context(self, query, top_n=5):
    
        # Generate the query embedding
        query_embedding = self.embedder.get_embedding(query)
    
        while True:
            try:
                query_response = self.index.query(
                    namespace=self.namespace,
                    top_k=top_n,
                    include_values=False,
                    include_metadata=True,
                    vector=query_embedding,
                )
                break
    
            except PineconeProtocolError:
                print("Pinecone needs a moment....")
                time.sleep(3)
                continue
    
        # Retrieve metadata for the relevant embeddings
        context_chunks = [
            match["metadata"]["context"] for match in query_response["matches"]
        ]
    
        return context_chunks
    



                

        
