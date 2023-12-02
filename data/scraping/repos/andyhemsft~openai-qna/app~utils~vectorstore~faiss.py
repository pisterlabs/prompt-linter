import os
import logging
from typing import List, Optional, Dict, Any, Tuple
import faiss

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS as FAISS_TYPE
from langchain.vectorstores.base import VectorStoreRetriever

from app.utils.vectorstore.base import BaseVectorStore
from app.config import Config

logger = logging.getLogger(__name__)

class FAISSExtended(BaseVectorStore):
    """This class represents a FAISS Vector Store."""

    def __init__(self, config: Config, embeddings: Embeddings):
        """
        Initialize the FAISS Vector Store.

        Args:
            config: the config object
            embeddings: the embeddings model
        """

        super().__init__(config, embeddings)

        # initialize the vector store
        embedding_size = config.OPENAI_EMBEDDING_SIZE
        index = faiss.IndexFlatL2(embedding_size)
        embedding_fn = embeddings.embed_query
        self.vector_store  = FAISS_TYPE(embedding_fn, index, InMemoryDocstore({}), {})

        # texts = ["FAISS"]
        # self.vector_store = FAISS_TYPE.from_texts(texts, embeddings)

    def load_local(self, file_path: str) -> None:
        """This function loads the vector store from a local file."""

        # Check if the file exists
        if os.path.exists(file_path):
            logger.info(f"FAISS local file '{file_path}' exists, loading it")
            self.vector_store = FAISS_TYPE.load_local(file_path, self.embeddings)

        else:
            logger.warning(f"FAISS local file '{file_path}' does not exist, creating a new one")
            self.vector_store.save_local(file_path)

    def save_local(self, file_path: str) -> None:
        """This function saves the vector store to a local file."""

        self.vector_store.save_local(file_path)

    def add_documents(
            self, 
            documents: List[Document], 
            index_name: Optional[str] = None,
            **kwargs: Any
        ) -> None:
        """This function adds documents to the vector store.
        
        Args:
            documents: the documents to add
            index_name: the index name
        Returns:
            none
        """

        logger.warning('FAISS does not support index_name parameter')

        texts = [document.page_content for document in documents]
        metadatas = [document.metadata for document in documents]
        self.add_texts(texts=texts, metadatas=metadatas)

    def add_texts(
            self, 
            texts: List[str],
            metadatas: Optional[List[Dict[str, Any]]] = None, 
            index_name: str = None, 
            **kwargs: Any
        ) -> None:
        """This function adds texts to the vector store.
        
        Args:
            texts: the texts to add
            index_name: the index name
        Returns:
            none
        """
        # # Load the local file
        # self.load_local(self.config.FAISS_LOCAL_FILE_INDEX)
        self.vector_store.add_texts(texts, metadatas=metadatas)
        # # Save to local file
        # self.save_local(self.config.FAISS_LOCAL_FILE_INDEX)

    def similarity_search( 
            self, 
            query: str, 
            k: int = 4, 
            filter: Optional[Dict[str, Any]] = None,
            index_name: Optional[str] = None
        ) -> List[Tuple[Document, float]]:
        """This function performs a similarity search.
        
        Args:
            query: the query
            k: the number of results
            filter: the filter
        Returns:
            docs and relevance scores in the range [0, 1].
        """

        return self.vector_store.similarity_search_with_relevance_scores(query, k, filter=filter)

    def create_index(self, 
                     index_name: str, 
                     metadata_schema: Dict[str, str]=None, 
                     distance_metric: Optional[str]="COSINE"
                     ) -> None:
        """This function creates an index.
        
        Args:
            index_name: the index name
        Returns:
            none
        """

        logger.warning('FAISS does not support creating indexes')

    def drop_index(self, index_name: str) -> None:
        """This function drops an index.
        
        Args:
            index_name: the index name
        Returns:
            none
        """

        logger.warning('FAISS does not support dropping indexes')


    def get_retriever(self, index_name: Optional[str] = None) -> VectorStoreRetriever:
        """This function returns a retriever object."""

        logger.warning('FAISS does not support index name parameter')

        # Make sure we load it before using it
        self.load_local(self.config.FAISS_LOCAL_FILE_INDEX)
        return self.vector_store.as_retriever()
    
    def check_existing_index(self, index_name: str = None) -> bool:
        """This function checks if the index exists.
        
        Args:
            index_name: the index name
        Returns:

        """
            
        logger.warning('FAISS does not support index name parameter')

        return True

