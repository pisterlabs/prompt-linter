import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import cohere
import os


class ChromaPeek:
    def __init__(self, path):
        self.client = chromadb.PersistentClient(path)
        self.model = None
        self.co = None

    # function that returs all collection's name
    def get_collections(self):
        collections = []

        for i in self.client.list_collections():
            collections.append(i.name)
        return collections

    # function to return documents/ data inside the collection
    def get_collection_data(self, collection_name, dataframe=False):
        data = self.client.get_collection(name=collection_name).get(include=['metadatas', 'documents', 'uris'])
        if dataframe:
            return pd.DataFrame(data).drop(columns=['embeddings', 'data'])
        return data

    # function to query the selected collection
    def query(self, query_str, collection_name, k=5, dataframe=False):
        collection = self.client.get_collection(collection_name)
        if collection_name == 'images_clip' and self.model is None:
            self.model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
        elif collection_name == 'texts_cohere' and self.model is None:
            load_dotenv()
            self.co = cohere.Client(os.getenv('COHERE_API_KEY'))
        elif self.model is None:
            self.model = SentenceTransformer('intfloat/multilingual-e5-large')
        if collection_name == 'texts_cohere':
            embeddings = self.co.embed([query_str], input_type="search_query", model="embed-multilingual-v3.0").embeddings
        elif collection_name == 'images_clip':
            embeddings = self.model.encode([query_str], normalize_embeddings=True).tolist()
        else:
            text = f'query: {query_str}'
            embeddings = self.model.encode([text], normalize_embeddings=True).tolist()
        res = collection.query(
            query_embeddings=embeddings[0], n_results=min(k, len(collection.get())),
            include=['metadatas', 'uris', 'documents', 'distances']
        )
        out = {}
        for key, value in res.items():
            if value:
                out[key] = value[0]
            else:
                out[key] = value
        if dataframe:
            return pd.DataFrame(out).drop(columns=['embeddings', 'data'])
        return out
