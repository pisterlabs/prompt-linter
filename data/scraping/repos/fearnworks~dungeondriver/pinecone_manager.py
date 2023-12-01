import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from dataclasses import dataclass


@dataclass
class PineconeConfig:
    openai_key: str
    pinecone_key: str
    pinecone_env: str
    index_name: str


class PineconeSessionManager:
    """
    A class for managing Pinecone sessions and indexes.

    Attributes:
        embeddings (OpenAIEmbeddings): The embeddings object to use for indexing.
        index_name (str): The name of the Pinecone index to use.
        index (pinecone.GRPCIndex): The Pinecone index object.
        docsearch (Pinecone): The Pinecone search object.
    """

    def __init__(self, embeddings, config: PineconeConfig):
        """
        Initializes a new PineconeSessionManager instance.

        Args:
            embeddings (OpenAIEmbeddings): The embeddings object to use for indexing.
            index_name (str): The name of the Pinecone index to use.
        """
        self.embeddings = embeddings
        self.index_name = config.index_name
        # initialize pinecone
        pinecone.init(
            api_key=config.pinecone_key,  # find at app.pinecone.io
            environment=config.pinecone_env,  # next to api key in console
        )

        if self.index_name not in pinecone.list_indexes():
            raise ValueError(f"Index {self.index_name } not found")
        self.index = pinecone.GRPCIndex(self.index_name)
        self.docsearch: Pinecone = Pinecone.from_existing_index(
            index_name=self.index_name, embedding=embeddings
        )


def get_default_pinecone_session(config: PineconeConfig) -> PineconeSessionManager:
    """
    Returns a default PineconeSessionManager instance with OpenAI embeddings and a default index name.

    Returns:
        PineconeSessionManager: A new PineconeSessionManager instance.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=config.openai_key)
    pc_session = PineconeSessionManager(embeddings, config)
    return pc_session
