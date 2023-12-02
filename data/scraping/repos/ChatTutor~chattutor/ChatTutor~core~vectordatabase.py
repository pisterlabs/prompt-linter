import chromadb
from chromadb.utils import embedding_functions
from typing import List
from core.definitions import Text
import openai
import requests
import json
import os

# Setting up user and URL for activeloop
username = "mit.quantum.ai"
activeloop_url = "https://app.activeloop.ai/api/query/v1"


def embedding_function(texts, model="text-embedding-ada-002"):
    """Function to generate embeddings for given texts using OpenAI API

    Args:
        texts (List(Text)): texts to generate embeddings for
        model (str, optional): Model to use. Defaults to "text-embedding-ada-002".

    Returns:
        embeddings of the texts
    """
    if isinstance(texts, str):
        texts = [texts]
    texts = [t.replace("\n", " ") for t in texts]
    return [
        data["embedding"]
        for data in openai.Embedding.create(input=texts, model=model)["data"]
    ]


# Loading API keys from .env.yaml
# print('vectordb env variables:', os.environ)
if "CHATTUTOR_GCP" in os.environ or "_CHATUTOR_GCP" in os.environ:
    openai.api_key = os.environ["OPENAI_API_KEY"]
else:
    import yaml

    with open(".env.yaml") as f:
        yamlenv = yaml.safe_load(f)
    keys = yamlenv["env_variables"]
    print(keys)
    os.environ["OPENAI_API_KEY"] = keys["OPENAI_API_KEY"]


class VectorDatabase:
    """
    Object that aids the loading, updating and adding of data to
    a database using one a database provider 

    Attributes
    ----------
    path : str
        path of the folder containing the database
    db_provider : str
        provider of the database: \'chroma\'
    """

    __valid_db_providers = ["chroma"]

    def __raise_exception_if_not_valid_db_provider(self):
        if self.db_provider in self.__valid_db_providers:
            pass
        else:
            raise ValueError(f"{self.db_provider} not valid. valid providers are {self.__valid_db_providers}")

    def __init__(self, path, db_provider="chroma", hosted=True):
        self.path = path
        self.hosted = hosted
        self.db_provider = db_provider
        self.__raise_exception_if_not_valid_db_provider()        

    def init_db(self):
        """Initializing the database client if the provider is 'chroma'"""
        if self.db_provider != "chroma":
            return
        # self.client = chromadb.HttpClient(host='34.123.154.72', port=8000)
        if self.hosted:
            ip = self.path.split(":")[0]
            port = int(self.path.split(":")[1])
            self.client = chromadb.HttpClient(host=ip, port=port)
        else:
            self.client = chromadb.PersistentClient(path=self.path)

    def load_datasource(self, name):
        """Loading the appropriate data source based on the database provider"""
        if self.db_provider == "chroma":
            self.load_datasource_chroma(name)
        else:
            self.__raise_exception_if_not_valid_db_provider()
        
    def load_datasource_chroma(self, collection_name):
        """Initialize the datasource attribute for the chroma provided VectorDatabase object"""
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            model_name="text-embedding-ada-002"
        )
        self.datasource = self.client.get_or_create_collection(
            name=collection_name, embedding_function=openai_ef
        )

    def delete_datasource_chroma(self, collection_name):
        collections = self.client.list_collections()
        coll_names = [coll.name for coll in collections]
        print(coll_names, collection_name)
        if collection_name in coll_names:
            self.client.delete_collection(name=collection_name)
            coll_names = [coll.name for coll in collections]
            print(coll_names, collection_name)

    def add_texts(self, texts: List[Text]):
        """Adding texts to the database based on the database provider

        Args:
            texts (List[Text]) : Texts to add to database
        """
        if self.db_provider == "chroma":
            self.add_texts_chroma(texts)
        else:
            self.__raise_exception_if_not_valid_db_provider()

    def add_texts_chroma(self, texts: List[Text]):
        """Adding texts to Chroma data source with specified ids, metadatas, and documents

        Args:
            texts (List[Text]): Texts to add to database
        """
        count = self.datasource.count()
        ids = [str(i) for i in range(count, count + len(texts))]
        print("ids:", ids)
        print("texts", texts)
        print(texts[0].doc.docname)
        self.datasource.add(
            ids=ids,
            metadatas=[{"doc": text.doc.docname} for text in texts],
            documents=[text.text for text in texts],
        )

    def query(self, prompt, n_results, from_doc, metadatas=False, distances=False):
        """Querying the database based on the database provider

        Args:
            prompt (string) : Query for the database
            n_results (int) : Number of results
            from_doc (Doc) : Select only lines where doc = from_doc
        """
        if self.db_provider == "chroma":
            data = self.query_chroma(
                prompt,
                n_results,
                from_doc,
                include=["documents", "metadatas", "distances"],
            )
            if metadatas:
                return (
                    data["documents"][0],
                    data["metadatas"][0],
                    data["distances"][0],
                    " ".join(data["documents"][0]),
                )
            return " ".join(data["documents"][0])
        else:
            self.__raise_exception_if_not_valid_db_provider()

    def query_chroma(self, prompt, n_results, from_doc, include=["documents"]):
        """Querying Chroma data source with specified query_texts, n_results, and optional where clause"""
        if from_doc:
            return self.datasource.query(
                query_texts=prompt,
                n_results=n_results,
                where={"doc": from_doc},
                include=include,
            )
        else:
            return self.datasource.query(
                query_texts=prompt, n_results=n_results, include=include
            )
