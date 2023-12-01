import json
from langchain.vectorstores import Chroma
from embeddings import EmbeddingSelector, EmbeddingFunction
from typing import List, Dict, Union
from loaders import LoaderSelector, Loader
from splitters import SplitterSelector, Splitter
from pydantic import BaseModel


class Vectorstore(BaseModel):
    name: str
    db: object


class DBListItem(BaseModel):
    name: str
    DBListItem: Dict[str, object] = {}


class VectorStoreList(BaseModel):
    vectorstore_dicts: List[DBListItem] = []
    vectorstore_path: str = "docs/vectorstores.json"


class Vectorstores:
    def __init__(self):
        self.embedding: EmbeddingFunction
        self.splitter: Splitter
        self.loader: Loader
        self.name: str
        self.current_collection: str
        self.vectordb = None
        self.vectorstore_list = VectorStoreList()

    def register_vectorstore(
        self, dbname, vectorstore, vectorstore_path="docs/vectorstores.json"
    ) -> None:
        self.vectorstore_path = vectorstore_path
        with open(self.vectorstore_path, "r") as file:
            vectordb_registar = json.loads(file.read())
            vectordb_registar[dbname] = vectorstore
        with open(self.vectorstore_path, "w") as file:
            file.write(json.dumps(vectordb_registar))
        self.vectorstore_list.vectorstore_dicts.append(
            DBListItem(name=dbname, DBListItem={dbname: vectorstore})
        )
        return "Vectorstore registered"

    def selected_vectorstore(self, dbname="chroma"):
        for db_list_item in self.vectorstore_list.vectorstore_dicts:
            if db_list_item.name == dbname:
                self.vectordb = db_list_item.DBListItem
                break
        else:
            raise Exception(f"Vectorstore with name {dbname} not found")


class ChromaDB(Vectorstores):
    def __init__(
        self,
        splitter_name="sentence_transformers",
        embedding_name="sentence_transformers",
        persistent_directory="docs",
        dbname="chroma",
    ):
        super().__init__()
        self.name = dbname
        self.splitter = SplitterSelector.select(splitter_name)
        self.embedding = EmbeddingSelector.select(embedding_name)
        self.persistent_directory = persistent_directory
        self.current_collection = dbname
        self.selected_vectorstore(dbname)
        self.vectordb = Chroma(
            splitter=self.splitter,
            embedding=self.embedding,
            persistent_directory=self.current_collection,
            dbname=self.current_collection,
        )
