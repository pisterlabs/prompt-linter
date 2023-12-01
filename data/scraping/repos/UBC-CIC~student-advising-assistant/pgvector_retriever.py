from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema import Document
from langchain.vectorstores.pgvector import PGVector, DistanceStrategy
from typing import List, Dict, Tuple, Callable, Optional
import os
import copy
import ast
from .tools import load_json_file
from .base import Retriever, INDEX_PATH

class MyPGVectorRetriever(PGVector):
    """
    Temporary fix since the current Langchain PGVector class has a bug so that it does not work
    with similarity score thresholds.
    https://github.com/langchain-ai/langchain/blob/v0.0.235/langchain/vectorstores/pgvector.py#L582
    """
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Add kwargs to support similarity search with threshold, since the threshold
        is a kwarg used by functions upstream
        """
        return super().similarity_search_with_score(query,k,filter)
    
    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Add kwargs to support similarity search with threshold, since the threshold
        is a kwarg used by functions upstream
        """
        result = super().similarity_search_with_score_by_vector(embedding,k,filter)
        print([score for doc, score in result])
        return result
    
    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        # Default strategy is to rely on distance strategy provided
        # in vectorstore constructor
        if self._distance_strategy == DistanceStrategy.COSINE:
            return self._cosine_relevance_score_fn
        elif self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._euclidean_relevance_score_fn
        elif self._distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self.distance_strategy}."
                "Consider providing relevance_score_fn to PGVector constructor."
            )
    
class PGVectorRetriever(Retriever):
    index_type: str = 'pgvector'
    index_config_path: str = os.path.join(INDEX_PATH, index_type, 'index_config.json')
    
    # Parameters from the program_info to use in the metadata filter for queries
    filter_params: List[str]
    # Maximum number of documents to return
    k: int
    
    def __init__(self, connection_string: str, verbose: bool = False):
        """
        Initialize the RDS PGvector retriever
        - connection_string: connection string for the pgvector DB
                             can be created using PGVector.connection_string_from_db_params
        - verbose: set retriever to verbose mode
        """
        super().__init__(verbose)
        
        # Load the config file
        index_config = load_json_file(self.index_config_path)
        
        # Load the dense embedding model
        embeddings_model = self._embeddings_model_from_config(index_config)
        self.num_embed_concats = len(index_config['embeddings'])
        
        # Connect to the pgvector db
        db = MyPGVectorRetriever.from_existing_index(embeddings_model, index_config['name'], connection_string=connection_string)
        
        self.retriever = VectorStoreRetriever(vectorstore=db)
            
    def semantic_search(self, filter: Dict, program_info: Dict, topic: str, query: str, k = 5, threshold = 0) -> List[Document]:
        """
        Return the documents from similarity search with the given context and query
        - filter: Dict of metadata filter keys and values
        - program_info: Dict of program information
        - topic: keyword topic of the query
        - query: the full query question
        - k: number of documents to return
        - threshold: relevance threshold, all returned documents must surpass the threshold
                     relevance is cosine-similarity based, so ranges between 0 and 1
                     larger scores indicate greater relevance
        """
        self.set_top_k(k)
        query_str, kwargs = self._query_converter(filter,program_info,topic,query)
        
        if threshold > 0:
            self.retriever.search_type = "similarity_score_threshold"
            self.retriever.search_kwargs['score_threshold'] = threshold
        else:
            self.retriever.search_type = 'similarity'
            
        self._output_query_verbose(query_str, self.retriever.search_kwargs)
        docs = self.retriever.get_relevant_documents(query_str,**kwargs)
        return self._response_converter(docs)
    
    def docs_from_ids(self, doc_ids: List[int]) -> List[Document]:
        """
        Return a list of documents from a list of document indexes
        """
        docs = self.retriever.fetch_by_id(doc_ids, self.namespace)
        return self._response_converter(docs)
    
    def set_top_k(self, k: int):
        """
        Set the retriever's 'top k' parameter, determines how many
        documents to return from semantic search
        """
        self.retriever.search_kwargs['k'] = k
        
    def _query_converter(self, filter: Dict, program_info: Dict, topic: str, query: str) -> Tuple[str,Dict]:
        """
        Generates a text query and keyword args for the retriever from the input
        - filter: Dict of metadata filter keys and values
        - program_info: Dict of program information
        - topic: keyword topic of the query
        - query: the full query question
        Returns:
        - Tuple of the query string, and Dict of kwargs to pass to the retriever
        """
        self.retriever.search_kwargs['filter'] = filter
        
        query_str = ' : '.join([value for value in list(program_info.values()) + [topic,query] if len(value) > 0])
        return self._retriever_combined_query(query_str), {}
    
    def _response_converter(self, response: List[Document]) -> List[Document]:
        """
        Decode the document metadatas from rds
        Since rds requires primitive datatypes for metadatas,
        evaluates strings into dicts/arrays
        """
        decode_columns = ['titles','parent_titles','links']
        for doc in response:
            for column in decode_columns:
                doc.metadata[column] = ast.literal_eval(doc.metadata[column])
            doc.page_content = doc.metadata['text']

        return response