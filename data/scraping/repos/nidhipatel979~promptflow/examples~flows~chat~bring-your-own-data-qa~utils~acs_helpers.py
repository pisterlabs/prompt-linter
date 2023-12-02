# Helper classes to load data into ACS and to also query from ACS 
# Similar to milvus_helpers.py

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
# from langchain.pydantic_v1 import root_validator
from langchain.schema import BaseRetriever
from langchain.utils import get_from_env
from langchain.vectorstores.base import VectorStore
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (  
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    PrioritizedFields,  
    SemanticField,  
    SearchField,  
    SemanticSettings,  
    VectorSearch,  
    ScoringProfile,
    HnswVectorSearchAlgorithmConfiguration,  
    SearchAlias
)  
import os
import json
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)
from datetime import datetime
from utils.config import Config
import uuid 
import copy 
import logging 
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)


class ACSHelper:
    """
    A utility class for interacting with Azure Cognitive Search.

    Args:
    - connection_args (dict): A dictionary containing the connection arguments for the Azure Cognitive Search service.

    Attributes:
    - connection_args (dict): A dictionary containing the connection arguments for the Azure Cognitive Search service.

    Methods:
        __init__(self, connection_args): Initializes the AcsHelpers class and checks the connection to the Azure Cognitive Search service.
    """

    def __init__(self, 
                 connection_args: Optional[Dict] = None) -> None:
        if connection_args is None:
            self.service_endpoint = Config.fetch('azure-search-service-endpoint')
            self.key = Config.fetch('azure-search-admin-key')
            self.connection_args = {
                'service_endpoint': self.service_endpoint,
                'key': self.key,
            }

        else:
            if 'key' not in connection_args or 'service_endpoint' not in connection_args:
                raise Exception("connection_args must include 'key' and 'service_endpoint' keys.")
            else:
                self.connection_args = connection_args
        try:
            self.search_index_client = SearchIndexClient(endpoint=self.connection_args['service_endpoint'], 
                                                        credential=AzureKeyCredential(self.connection_args['key'])
                                                        )
            print(f"Checking connection to {self.connection_args['service_endpoint']}")

            try:
                # Check the connection by getting the service statistics
                service_stats = self.search_index_client.get_service_statistics()
                print(f"Connection to {self.connection_args['service_endpoint']} successful.")
                print(service_stats)
            except Exception as e:
                print("Error checking connection to Azure Cognitive Search:", e)

        except Exception as e:
            print("Error connecting to Azure Cognitive Search:", e)
            
    def list_indexes(self):
        """
        Returns a list of all index names in the search index client.
        """
        indexes = [index for index in self.search_index_client.list_index_names()]
        return indexes

    def get_index(self,
                  index_name: str):
        """
        Returns the specified index if it exists, otherwise returns None.

        Args:
            index_name (str): The name of the index to retrieve.

        Returns:
            Index: The specified index if it exists, otherwise None.
        """
        if index_name in self.list_indexes():
            index = self.search_index_client.get_index(name=index_name)
            return index
        else:
            print(f"Specified index does not exist ")
            return None

    def get_schema(self,
                   index_name : str):
        """
        Get the schema of an index.

        Args:
            index_name (str): The name of the index.

        Returns:
            dict: A dictionary containing the fields of the index.

        Raises:
            Exception: If an error occurs while fetching the schema.
        """
        try:
            index = self.get_index(index_name)
            if index:
                return index.as_dict()["fields"]
        except Exception as e:
            print(f"an error occured while fetching the schema : {e}")
            
    
    def get_field_names(self,
                        index_name: str):
        """
        Returns a list of field names for a given index name.

        Args:
        index_name (str): The name of the index.

        Returns:
        list: A list of field names for the given index name.
        """
        try:
            schema = self.get_schema(index_name)
            return [fields["name"] for fields in schema] if schema else []
        except Exception as e:
            print(f"an error occured while fetching the fields : {e}")

    def create_vector_search_config(self, 
                                    name : str, 
                                    m: int = 4, 
                                    efConstruction: int =400, 
                                    efSearch: int = 500, 
                                    metric: str = "cosine",
                                    **kwargs: Any
                                    ):
        """
        Creates a VectorSearch configuration object with an HNSW algorithm configuration.

        Args:
            name (str): The name of the algorithm configuration.
            m (int, optional): The number of bi-directional links created for each new element during construction. Defaults to 4.
            efConstruction (int, optional): The maximum number of elements to visit during the construction of the search index. Defaults to 400.
            efSearch (int, optional): The maximum number of elements to visit during a search. Defaults to 500.
            metric (str, optional): The distance metric to use. Defaults to "cosine".
            **kwargs: Additional keyword arguments.

        Returns:
            VectorSearch: A VectorSearch configuration object.
        
        Sources: https://learn.microsoft.com/en-us/python/api/azure-search-documents/azure.search.documents.indexes.models.hnswvectorsearchalgorithmconfiguration?view=azure-python-preview

        """
        vector_config = VectorSearch(
                                    algorithm_configurations=[
                                        HnswVectorSearchAlgorithmConfiguration(
                                            name=name,
                                            kind="hnsw",
                                            parameters={
                                                "m": m,
                                                "efConstruction": efConstruction,
                                                "efSearch": efSearch,
                                                "metric": metric
                                            }
                                        )
                                    ]
                                )
        
        return vector_config
    

    def create_semantic_config(self, 
                                name : str, 
                                title_field_name: Optional[str] = None, 
                                prioritized_keywords_fields: Optional[List[str]]= [],  
                                prioritized_content_fields: Optional[List[str]] = [],
                                **kwargs: Any
                                ):
        """
        Creates a SemanticConfiguration object with the given parameters.

        Args:
            name (str): The name of the SemanticConfiguration object.
            title_field_name (Optional[str], optional): The name of the title field. Defaults to None.
            prioritized_keywords_fields (Optional[List[str]], optional): A list of prioritized keyword fields. Defaults to [].
            prioritized_content_fields (Optional[List[str]], optional): A list of prioritized content fields. Defaults to [].
            **kwargs (Any): Additional keyword arguments.

        Returns:
            SemanticConfiguration: The created SemanticConfiguration object.
        
        Sources: 
        https://learn.microsoft.com/en-us/azure/search/semantic-how-to-query-request?tabs=portal%2Cportal-query#1---choose-a-client 
        https://learn.microsoft.com/en-us/azure/search/semantic-search-overview 
        """
        semantic_config = SemanticConfiguration(  
                                            name=name,
                                            prioritized_fields=PrioritizedFields(
                                                title_field=SemanticField(field_name=title_field_name),
                                                prioritized_keywords_fields=[SemanticField(field_name=field) for field in prioritized_keywords_fields],
                                                prioritized_content_fields=[SemanticField(field_name=field) for field in prioritized_content_fields]
                                            )
                                        )
        print(semantic_config.as_dict())
        return semantic_config
    
    def create_fields(self, 
                      input_doc: Document, 
                      vector_search_config_name : str
                    ):
        """
        Create a list of fields to push to Azure Cognitive Search.

        Args:
        input_doc (Document): The document to extract metadata from.
        vector_search_config_name (str): The name of the vector search configuration.

        Returns:
        List[SearchField]: A list of fields to push to Azure Cognitive Search.
        """
        # create a list of fields to push
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=False, filterable=False, facetable=False),
            SearchableField(name="text", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_configuration=vector_search_config_name)
        ]
        meta_keys = list(input_doc.metadata.keys())


        for k in meta_keys:
            md = input_doc.metadata
            if "date" in k or type(md[k]) is datetime:
                fields.append(
                    SimpleField(name=f"{k}", type=SearchFieldDataType.DateTimeOffset, filterable=True)
                )
            else:
                if type(md[k]) is list:
                    initial_value = md[k][0]
                    if type(initial_value) is bool:
                        fields.append(
                            SearchField(name=f"{k}", type=SearchFieldDataType.Collection(SearchFieldDataType.Boolean), filterable=True)
                        )
                    else:
                        if type(initial_value) is float:
                            dtype = SearchFieldDataType.Double
                            fields.append(
                                SearchField(name=f"{k}", type=SearchFieldDataType.Collection(SearchFieldDataType.Double), filterable=True)
                            )
                        elif type(initial_value) is int:
                            dtype = SearchFieldDataType.Int32
                            fields.append(
                                SearchField(name=f"{k}", type=SearchFieldDataType.Collection(SearchFieldDataType.Int32), filterable=True)
                            )
                        else:
                            dtype = SearchFieldDataType.String
                            fields.append(
                                SearchableField(name=f"{k}", type=dtype, filterable=True, collection=True)
                            )
                elif type(md[k]) is bool:
                    fields.append(
                        SimpleField(name=f"{k}", type=SearchFieldDataType.Boolean, filterable=True)
                    )
                else:
                    if type(md[k]) is float:
                        dtype = SearchFieldDataType.Double
                        fields.append(
                            SimpleField(name=f"{k}", type=dtype, filterable=True)
                        )
                    elif type(md[k]) is int:
                        dtype = SearchFieldDataType.Int32
                        fields.append(
                            SimpleField(name=f"{k}", type=dtype, filterable=True)
                        )
                    else:
                        dtype = SearchFieldDataType.String
                        fields.append(
                            SearchableField(name=f"{k}", type=dtype, filterable=True)
                        )
        # for field in fields:
        #     print(field.as_dict())
        return fields

    def create_index(self, 
                     index_name, 
                     fields, 
                     scoring_profile=None, 
                     cors_options=None, 
                     semantic_config=None, 
                     vector_search_config=None):
        """
        Creates an index in Azure Cognitive Search with the given parameters.

        Args:
            index_name (str): The name of the index to be created.
            fields (list): A list of field objects that define the schema for the index.
            scoring_profile (list, optional): A list of scoring profile objects that define the scoring behavior for the index. Defaults to None.
            cors_options (CorsOptions, optional): An object that defines the Cross-Origin Resource Sharing (CORS) options for the index. Defaults to None.
            semantic_config (SemanticSearchConfig, optional): An object that defines the semantic search configuration for the index. Defaults to None.
            vector_search_config (VectorSearchConfig, optional): An object that defines the vector search configuration for the index. Defaults to None.

        Returns:
            Index: The index object that was created in Azure Cognitive Search.
        """
        semantic_settings = None

        if semantic_config is not None:
            semantic_settings = SemanticSettings(configurations=[semantic_config])

        # Create the index object
        search_index = SearchIndex(
                                name=index_name,
                                fields=fields,
                                scoring_profiles=scoring_profile,
                                cors_options=cors_options,
                                semantic_settings=semantic_settings,
                                vector_search=vector_search_config
                            )

        # Create the index in Azure Cognitive Search
        result = self.search_index_client.create_or_update_index(search_index)
        print(f' {result.name} created')
        return result
    
    def delete_index(self, index_name):
        """
        Deletes an Azure Cognitive Search index with the given name.

        Args:
        - index_name (str): The name of the index to delete.

        Returns:
            None
        """

        self.search_index_client.delete_index(index_name)
        print(f' {index_name} deleted')

    def get_search_client(self, index_name):
        """
        Returns a SearchClient object for the given index name using the connection arguments stored in self.connection_args.

        Args:
        - index_name (str): The name of the index to create the SearchClient for.

        Returns:
        - search_client (SearchClient): A SearchClient object for the given index name.
        """

        return SearchClient(endpoint=self.connection_args['service_endpoint'], 
                            credential=AzureKeyCredential(self.connection_args['key']),
                            index_name=index_name
                            )
    
    def batch_upload_json_data_to_index(self, 
                                        json_data: List[dict],
                                        search_client : SearchClient, 
                                        batch_size: int =500
                                    ):
        """
        Batch uploads JSON data to an Azure Cognitive Search index using the provided client.

        Args:
        - json_data (list): A list of JSON objects to upload to the index.
        - client (SearchClient): An instance of the Azure Cognitive Search client.
        - batch_size (int, optional): The number of records to include in each batch. Defaults to 500.

        Returns:
            None
        """
        batch_array = []
        count = 0
        batch_counter = 0
        for i in json_data:
            count += 1
            batch_array.append(
                i
            )

            # In this sample, we limit batches to 500 records.
            # When the counter hits a number divisible by 500, the batch is sent.
            if count % batch_size == 0:
                search_client.upload_documents(documents=batch_array)
                search_client.delete_documents
                batch_counter += 1
                print(f"Batch sent! - #{batch_counter}")
                batch_array = []

        # This will catch any records left over, when not divisible by 1000
        if len(batch_array) > 0:
            search_client.upload_documents(documents=batch_array)
            batch_counter += 1
            print(f"Final batch sent! - #{batch_counter}")

        print("Done!")

    def create_or_update_alias(self, alias_name, index_name):
        """
        Creates or updates an alias for a given index in Azure Cognitive Search.

        Args:
        - alias_name (str): The name of the alias to create or update.
        - index_name (str): The name of the index to associate with the alias.

        Returns:
        None
        """

        alias = SearchAlias(name=alias_name, indexes=[index_name])
        self.search_index_client.create_or_update_alias(alias)
        print(f"Alias {alias_name} updated for {index_name}!")

    
    def append_prefix_to_metadata(self, docs, prefix='metadata_'):
        """
        Appends a prefix to all keys in the metadata dictionary of each document in the given list of documents.

        Args:
            docs (list): A list of documents.
            prefix (str, optional): The prefix to append to each key in the metadata dictionary. Defaults to 'metadata_'.

        Returns:
            list: The updated list of documents.
        """
        for doc in docs:
            doc.metadata = {prefix + key: value for key, value in doc.metadata.items()}
        return docs
    

    def document_pre_processing(self, 
                                docs: List[Document], 
                                embeddings_list: List[float],
                                prefix: str ='metadata_') -> List[Dict]:
        """
        Pre-processes a list of Document objects for uploading to an Azure Cognitive Search index.

        Args:
        - docs (List[Document]): A list of Document objects to be pre-processed.
        - embeddings_list (List[float]): A list of embeddings corresponding to each Document object.
        - prefix (str, optional): A prefix to be added to each metadata key. Defaults to 'metadata_'.

        Returns:
        - input_data (List[Dict]): A list of dictionaries containing pre-processed data for uploading to an Azure Cognitive Search index.
        - docs (List[Document]): The original list of Document objects, with metadata keys updated to include the specified prefix.

        """
        docs = self.append_prefix_to_metadata(docs=docs, prefix=prefix)

        input_data = []
        for i, doc in enumerate(docs):

            input_dict = {}
            input_dict['id'] = str(uuid.uuid4())
            input_dict['text'] = doc.page_content
            input_dict['embedding'] = embeddings_list[i]
            input_dict = {**input_dict, **doc.metadata}

            # convert metadata datetime to string in input_dict
            for key, value in input_dict.items():
                if isinstance(value, datetime):
                    input_dict[key] = value.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            input_data.append(input_dict)
        return input_data, docs


    def from_documents(
        self,
        documents: List[Document],
        embedding: Embeddings,
        index_name: str,
        index_alias_name: Optional[str] = None,
        vector_config_name: Optional[str] = None,
        semantic_config_name: Optional[str] = None,
        embeddings_list: Optional[List[float]] = None,
        **kwargs: Any,
    ):
        """
        Indexes a list of documents into Azure Cognitive Search service.

        Args:
            documents (List[Document]): A list of Document objects to be indexed.
            embedding (Embeddings): An Embeddings object to embed the documents.
            index_name (str): The name of the index to be created.
            index_alias_name (Optional[str], optional): The name of the index alias. Defaults to None.
            vector_config_name (Optional[str], optional): The name of the vector search configuration. Defaults to None.
            semantic_config_name (Optional[str], optional): The name of the semantic search configuration. Defaults to None.
            embeddings_list (Optional[List[float]], optional): A list of embeddings to be used instead of embedding the documents. Defaults to None.
            **kwargs (Any): Additional keyword arguments to be passed to the function.

        Returns:
            None
        """
        documents = copy.deepcopy(documents)
        # If embeddings_list is not provided, embed the documents
        texts = [d.page_content for d in documents]
        if embeddings_list is None:
            print('Embedding documents')
            try:
                embeddings_list = embedding.embed_documents(
                    list(texts)
                )
            except NotImplementedError:
                embeddings_list = [
                    embedding.embed_query(x) for x in texts
                ]
        else:
            print('Using provided embeddings')

        if vector_config_name is None:
            vector_config_name = f"{index_name}-vector-config"
        if semantic_config_name is None:
            semantic_config_name = f"{index_name}-semantic-config"

        # Create the vector search configuration
        vector_config = self.create_vector_search_config(name=vector_config_name,**kwargs)
        
        # Get list of dict from docs (input data)
        input_data, processed_documents = self.document_pre_processing(docs=documents,
                                                                        embeddings_list=embeddings_list,
                                                                        prefix='metadata_'
                                                                        )
        
        # Create the fields 
        fields = self.create_fields(input_doc=processed_documents[0],
                                    vector_search_config_name=vector_config_name)

        # Create the semantic search configuration
        # Get kwargs for semantic config and pass it to create_semantic_config
        # keywords fields will be metadata fields (append metadata_ prefix)
        # content fields will be text 
        if kwargs.get("title_field_name", None) is not None:
            title_field_name = "metadata_" + kwargs.get("title_field_name")
        else:
            title_field_name = None 
        semantic_config = self.create_semantic_config(name=semantic_config_name, 
                                                      title_field_name=title_field_name, 
                                                      prioritized_keywords_fields=["metadata_" + field for field in kwargs.get("prioritized_keywords_fields", [])],
                                                      prioritized_content_fields=kwargs.get("prioritized_content_fields", ["text"])
                                                    )
        
        # Get list of indexes
        index_name_list = list(self.search_index_client.list_index_names())
        if index_name in index_name_list:
            print(f"Index {index_name} already exists. Deleting index...")
            # Delete the index if it already exists
            self.search_index_client.delete_index(index=index_name)

        # Create the index
        self.create_index(index_name=index_name, 
                          fields=fields, 
                          semantic_config=semantic_config, 
                          vector_search_config=vector_config
                        )

        # Get the Search Client
        search_client = self.get_search_client( index_name=index_name)

        # Upload the data to the index
        self.batch_upload_json_data_to_index(json_data=input_data, 
                                             search_client=search_client
                                            )

        # Update alias name if provided
        if index_alias_name is not None:
            self.create_or_update_alias(alias_name=index_alias_name, 
                                        index_name=index_name
                                    )


class AIAAzureSearch(VectorStore):
    """`Azure Cognitive Search` vector store."""
    def __init__(
        self,
        azure_search_endpoint: str,
        azure_search_key: str,
        index_name: str,
        embedding_function: Callable,
        search_type: str = "similarity",
        semantic_configuration_name: Optional[str] = None,
        semantic_query_language: str = "en-us",
        fields: Optional[List[SearchField]] = None,
        # vector_search: Optional[VectorSearch] = None,
        semantic_settings: Optional[SemanticSettings] = None,
        scoring_profiles: Optional[List[ScoringProfile]] = None,
        default_scoring_profile: Optional[str] = None,
        acs_helper : Optional[Callable] = None, 
        **kwargs: Any,
    ):
        super().__init__()
        credential = AzureKeyCredential(azure_search_key)
        self.searchclient =  SearchClient(
                endpoint=azure_search_endpoint,
                index_name=index_name,
                credential=credential,
            )
        self.embedding_function = embedding_function
        self.index_name = index_name
        self.search_type = search_type
        self.semantic_query_language = semantic_query_language
        self.semantic_configuration_name = semantic_configuration_name
        if acs_helper is not None:
            self.ah = acs_helper  
        else: 
            connection_args = {
                'service_endpoint': azure_search_endpoint,
                'key': azure_search_key,
            }
            self.ah = ACSHelper(connection_args=connection_args)

    def get_index(self):
        return self.ah.get_index(self.index_name)
    
    def get_schema(self):
        return self.ah.get_schema(self.index_name)
    
    def get_field_names(self):
        return self.ah.get_field_names(self.index_name)
    
    def query_collection(self, expr, offset=0, k=100, output_fields=[], consistency_level='Strong'):
        # Need to specify the fields you want returned in output_fields, otherwise will only return pk
        # expr is the search expression, see: https://milvus.io/docs/boolean.md
        # return self.ah.query_collection(self.collection_name, expr, offset, k, output_fields, consistency_level)
        pass # Todo
    
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts data to an existing index."""
        
    def similarity_search(
        self, query: str, k: int =3, **kwargs: Any
    ) -> List[Document]:
        search_type = kwargs.get("search_type", self.search_type)
        if search_type == "similarity":
            docs = self.vector_search(query, k=k, **kwargs)
        elif search_type == "hybrid":
            docs = self.hybrid_search(query, k=k, **kwargs)
        elif search_type == "semantic_hybrid":
            docs = self.semantic_hybrid_search(query, k=k, **kwargs)
        elif search_type == "Cross-Field Vector Search":
            pass # TODO
        else:
            raise ValueError(f"search_type of {search_type} not allowed.")
        return docs

    def vector_search(self, query: str, k: int = 3, **kwargs: Any) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 3.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = self.vector_search_with_score(
            query,k=k,
            # multi_vector_search=kwargs.get("multi_vector_search", False), 
            # filters=kwargs.get("filters", None),selects=kwargs.get("selects", None),
            **kwargs
        )
        for doc in docs_and_scores:
            doc[0].metadata['similarity_score'] = doc[1]
        
        return [doc for doc, _ in docs_and_scores]

    def vector_search_with_score(
        self, query: str,
        # vector_fields: List[str], #TODO: see if we can extract embeddig field through API
        # FIELDS_CONTENT : str,
        # FIELDS_METADATA :List[str],
        k: int = 3,
        multi_vector_search :bool = False,
        filters: Optional[str] = None , 
        selects:Optional[List[str]] =None,**kwargs) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            vector_fields: Provides the list of fields that you want to search the query against.
            k: Number of Documents to return. Defaults to 3.
            multi_vector_search: True if you want to search against multiple vectors, False if you want to search against a single vector
            filters : filter to apply to the query. e.g : "metadata_language eq 'en'"
            selects : fields to return in the query. e.g : ["metadata_language","metadata_title"]
        Returns:
            List of Documents most similar to the query and score for each
        """
        from azure.search.documents.models import Vector
        # vector_fields = kwargs["vector_fields"]
        
        
        if multi_vector_search:
            FIELDS_VECTORS = [schema["name"] for schema in self.get_schema(index_name=self.index_name) if schema["type"]=='Collection(Edm.Single)']
            final_vectors_list = FIELDS_VECTORS
        else:
            final_vectors_list = ["embedding"]
        
        # print(vectors_list)
        vectors_list = []
        for vectors_field_name in final_vectors_list:
            vectors_list.append(Vector(value=self.embedding_function.embed_query(query),
                                k=k, 
                                fields=vectors_field_name))
        results = self.searchclient.search(
            search_text=None,
            vectors=vectors_list,
            select=selects,
            filter=filters,
            top=k
        )
        # Convert results to Document objects
        
        docs = []
        # each search vectore
        for res in results:
            meta = {key: value for key, value in res.items() if key.startswith('metadata')}
            docs.append(
                (
                    Document(
                        page_content=res["text"], metadata=meta
                    ),
                    res["@search.score"],
                )
            )
        
        return docs

    

    def hybrid_search(self, query: str, k: int = 3, **kwargs: Any) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 3.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = self.hybrid_search_with_score(
            query, k=k,**kwargs)
        for doc in docs_and_scores:
            doc[0].metadata['similarity_score'] = doc[1]
        return [doc for doc, _ in docs_and_scores]

    def hybrid_search_with_score(
        self, query: str, k: int = 3,
        multi_vector_search :bool =False, filters: Optional[str] = None , 
        selects:Optional[List[str]] =None,**kwargs
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query with an hybrid query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 3.
            multi_vector_search: True if you want to search against multiple vectors, False if you want to search against a single vector
            filters : filter to apply to the query. e.g : "metadata_language eq 'en'"
            selects : fields to return in the query. e.g : ["metadata_language","metadata_title"]
        Returns:
            List of Documents most similar to the query and score for each
        """
        from azure.search.documents.models import Vector

        if multi_vector_search:
            FIELDS_VECTORS = [schema["name"] for schema in self.get_schema(index_name=self.index_name) if schema["type"]=='Collection(Edm.Single)']
            final_vectors_list = FIELDS_VECTORS
        else:
            final_vectors_list = ["embedding"]
        
        # print(vectors_list)
        vectors_list = []
        for vectors_field_name in final_vectors_list:
            vectors_list.append(Vector(value=self.embedding_function.embed_query(query),
                                k=k, 
                                fields=vectors_field_name))
        results = self.searchclient.search(
            search_text=query,
            vectors=vectors_list,
            select=selects,
            filter=filters,
            top=k
        )
        # Convert results to Document objects
        docs = []
        # each search vectore
        for res in results:
            meta = {key: value for key, value in res.items() if key.startswith('metadata')}
            docs.append(
                (
                    Document(
                        page_content=res["text"], metadata=meta
                    ),
                    float(res["@search.score"]),
                )
            )
        
        return docs

    def semantic_hybrid_search(
        self, query: str, k: int = 3,
        **kwargs: Any
    ) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k: Number of Documents to return. Defaults to 3.
            multi_vector_search: True if you want to search against multiple vectors, False if you want to search against a single vector
            filters : filter to apply to the query. e.g : "metadata_language eq 'en'"
            selects : fields to return in the query. e.g : ["metadata_language","metadata_title"]
        
        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = self.semantic_hybrid_search_with_score(
            query, k=k, **kwargs
        )
        for doc in docs_and_scores:
            doc[0].metadata['similarity_score'] = doc[1]
        return [doc for doc, _ in docs_and_scores]
        
    def semantic_hybrid_search_with_score(
        self, query: str, k: int = 3,multi_vector_search :bool =False, filters: Optional[str] = None , 
        selects:Optional[List[str]] =None,**kwargs
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query with an hybrid query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 3.

        Returns:
            List of Documents most similar to the query and score for each
        """

        from azure.search.documents.models import Vector

        if multi_vector_search:
            FIELDS_VECTORS = [schema["name"] for schema in self.get_schema(index_name=self.index_name) if schema["type"]=='Collection(Edm.Single)']
            final_vectors_list = FIELDS_VECTORS
        else:
            final_vectors_list = ["embedding"]
        
        # print(vectors_list)
        vectors_list = []
        for vectors_field_name in final_vectors_list:
            vectors_list.append(Vector(value=self.embedding_function.embed_query(query),
                                k=50, 
                                fields=vectors_field_name))
        results = self.searchclient.search(
            search_text=query,
            vectors=vectors_list,
            select=selects,
            filter=filters,
            query_type="semantic",
            query_language=self.semantic_query_language,
            semantic_configuration_name=self.semantic_configuration_name,
            query_caption="extractive",
            query_answer="extractive",
            top=k,
        )
        ### Semantic Answers : https://learn.microsoft.com/en-us/azure/search/semantic-answers#prerequisites
        ### Semantic Answers : You can think of this as a search that will also respond to a query if query is like a question
        ## if there is sematic answer, it will be returned as follows:
        ###### 
        ##    {"key": "4123",
        ##    "text": "Sunlight heats the land all day, warming that moist air and causing it to rise high into the   atmosphere until it cools and condenses into water droplets. Clouds generally form where air is ascending (over land in this case),   but not where it is descending (over the river).",
        ##    "highlights": "Sunlight heats the land all day, warming that moist air and causing it to rise high into the   atmosphere until it cools and condenses into water droplets. Clouds generally form<em> where air is ascending</em> (over land in this case),   but not where it is<em> descending</em> (over the river).",
        ##    "score": 0.94639826}
        ###### 
        semantic_answers = results.get_answers() or []
        semantic_answers_dict: Dict = {}
        for semantic_answer in semantic_answers:
            semantic_answers_dict[semantic_answer.key] = {
                "text": semantic_answer.text,
                "highlights": semantic_answer.highlights,
            }

        # Convert results to Document objects
        docs = []
        # each search vectore
        for res in results:
            meta = {key: value for key, value in res.items() if key.startswith('metadata')}
            docs.append( 
                        (Document(
                            page_content=res["text"],
                            metadata={
                                **meta,
                                **{
                                    "captions": {
                                        "text": res.get("@search.captions", [{}])[0].text,
                                        "highlights": res.get("@search.captions", [{}])[
                                            0
                                        ].highlights,
                                    }
                                    if res.get("@search.captions")
                                    else {},
                                    "answers": semantic_answers_dict,
                                },
                                **{"search_reranker_score":float(res["@search.reranker_score"])}
                            },
                        ),
                        float(res["@search.score"]),
                    ))
        
        return docs
        
        # Get Semantic Answers
        
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        azure_search_endpoint: str = "",
        azure_search_key: str = "",
        index_name: str = "langchain-index",
        **kwargs: Any,
    ) -> VectorStore:
        # Creating a new Azure Search instance
        azure_search = cls(
            azure_search_endpoint,
            azure_search_key,
            index_name,
            embedding.embed_query,
        )
        azure_search.add_texts(texts, metadatas, **kwargs)
        return azure_search