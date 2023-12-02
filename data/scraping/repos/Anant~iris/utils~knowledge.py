"""Chain that calls data stored in a knowledge base created by me

"""
# Imports to connect to Cassandra
from cassandra.cluster import (
    Cluster,
)
from cassandra.auth import PlainTextAuthProvider
#from cqlsession import getCQLSession, getCQLKeyspace

# Imports to manage index data
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader, CSVLoader
from langchain.vectorstores.cassandra import Cassandra
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

import os
import json
import sys
from dotenv import load_dotenv

from typing import Any, Dict, Optional
from pydantic import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env

# Load the environment variables 
load_dotenv()
ASTRA_DB_TOKEN_BASED_USERNAME = os.environ["ASTRA_DB_TOKEN_BASED_USERNAME"]
ASTRA_DB_TOKEN_BASED_PASSWORD = os.environ["ASTRA_DB_TOKEN_BASED_PASSWORD"]
ASTRA_DB_SECURE_BUNDLE_PATH = os.environ["ASTRA_DB_SECURE_BUNDLE_PATH"]
ASTRA_DB_KEYSPACE = os.environ["ASTRA_DB_KEYSPACE"]

# Utility functions , likely don't really need this but these were in the Google Colab example 
def getCQLSession(mode='astra_db'):
    if mode == 'astra_db':
        cluster = Cluster(
            cloud={
                "secure_connect_bundle": ASTRA_DB_SECURE_BUNDLE_PATH,
            },
            auth_provider=PlainTextAuthProvider(
                ASTRA_DB_TOKEN_BASED_USERNAME,
                ASTRA_DB_TOKEN_BASED_PASSWORD,
            ),
        )
        astraSession = cluster.connect()
        return astraSession
    else:
        raise ValueError('Unsupported CQL Session mode')

def getCQLKeyspace(mode='astra_db'):
    if mode == 'astra_db':
        return ASTRA_DB_KEYSPACE
    else:
        raise ValueError('Unsupported CQL Session mode')

cqlMode = 'astra_db'
session = getCQLSession(mode=cqlMode)
keyspace = getCQLKeyspace(mode=cqlMode)

os.environ['OPENAI_API_TYPE'] = 'open_ai'
    
llm = OpenAI(temperature=0)
myEmbedding = OpenAIEmbeddings()


# Ingest data methods - this can likely be put into another all together but keeping it here for simplicity's sake

# Ingest the amontillado text 
def ingest_amontillado():
    table_name = 'text_amontillado'

    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=Cassandra,
        embedding=myEmbedding,
        text_splitter=CharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=0,
        ),
        vectorstore_kwargs={
            'session': session,
            'keyspace': keyspace,
            'table_name': table_name,
        },    
    )

    print(f"Loading data into Vector Store: {table_name}: Started")
    text_loader = TextLoader('data/documents/amontillado.txt', encoding='utf8')
    index = index_creator.from_loaders([text_loader])
    print(f"Loading data into Vector Store: {table_name}: Done")

# Ingest the failed bank list csv 
def ingest_banks():
    table_name = 'csv_banklist'

    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=Cassandra,
        embedding=myEmbedding,
        vectorstore_kwargs={
            'session': session,
            'keyspace': keyspace,
            'table_name': table_name,
        },    
    )

    print(f"Loading data into Vector Store: {table_name}: Started")
    csv_loader = CSVLoader('data/documents/banklist.csv', encoding='latin-1')
    index = index_creator.from_loaders([csv_loader])
    print(f"Loading data into Vector Store:  {table_name}: Done")


# reusable get index method to get the index backed by a Cassandra vector store 
def getIndex(name:str):
    table_name = name

    myCassandraVStore = Cassandra(
        embedding=myEmbedding,
        session=session,
        keyspace=keyspace,
        table_name= table_name
    )

    index = VectorStoreIndexWrapper(vectorstore=myCassandraVStore)
    return index


class HiddenPrints:
    """Context manager to hide prints."""

    def __enter__(self) -> None:
        """Open file to pipe stdout to."""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *_: Any) -> None:
        """Close file that stdout was piped to."""
        sys.stdout.close()
        sys.stdout = self._original_stdout

class KnowledgeWrapper(BaseModel):
    """Wrapper data in a Cassandra Vector Store

    To use, you should have the environment variables set in your .env file

    ASTRA_DB_TOKEN_BASED_USERNAME = 
    ASTRA_DB_TOKEN_BASED_PASSWORD = 
    ASTRA_DB_SECURE_BUNDLE_PATH = 
    ASTRA_DB_KEYSPACE = 

    Example:
        .. code-block:: python

            from utils.knowledge import KnowledgeWrapper
            knowledge = KnowledgeWrapper()
    """
    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that we can get the Keyspace and Creds via ENV Vars"""
        
        try:
            cqlMode = 'astra_db'
            session = getCQLSession(mode=cqlMode)
            keyspace = getCQLKeyspace(mode=cqlMode)
        except ImportError:
            raise ValueError(
                "Could not find environment variables for Astra"
                "Please add them to your .env file as instructed."
            )
        return values

    
    def banks(self, query: str) -> str:
        """Get answer from Vector Store with text of The Cask of Amontillado"""

        q = query  # str | Search query term or phrase.
        table_name = 'csv_banklist'
        index = getIndex(table_name)
        with HiddenPrints():
            try:
                # Get Answer
                response = index.query(query, llm=llm)            
            except ApiException as e:
                raise ValueError(f"Got error from Cassio / LangChain Index: {e}")

        return response

    def amontillado(self, query: str) -> str:
        """Get answer from Vector Store with Failed Banks"""

        q = query  # str | Search query term or phrase.
        table_name = 'text_amontillado'
        index = getIndex(table_name)

        with HiddenPrints():
            try:
                # Get Answer
                response = index.query(query, llm=llm)            
            except ApiException as e:
                raise ValueError(f"Got error from Cassio / LangChain Index: {e}")

        return response



# Check if the file is run directly
# Can run this with `python utils/knowledge.py` to load the data 
if __name__ == "__main__":
    # Execute the code to upload data
    ingest_amontillado()
    ingest_banks()