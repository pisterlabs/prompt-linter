# Ref: https://github.com/hwchase17/langchain/pull/3350
# 05/11/2023: Replace with langchain apis after the PR is merged

"""Vertex Matching Engine implementation of the vector store."""
from __future__ import annotations
import logging
import uuid
import json
from typing import Any, Iterable, List, Optional, Type
from requests import Response
# zeitghost
from zeitghost.vertex.Embeddings import VertexEmbeddings
from zeitghost.vertex.LLM import VertexLLM
from zeitghost.vertex.Helpers import map_prompt
from zeitghost.vertex.Helpers import combine_prompt
# LangChain stuff
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.vectorstores.base import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import YoutubeLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import BigQueryLoader
from langchain.document_loaders import GCSFileLoader
# GCP & Vertex
from google.cloud import storage
from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndex
from google.cloud.aiplatform import MatchingEngineIndexEndpoint
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import MatchNeighbor
from google.cloud import aiplatform_v1
from google.cloud import documentai
from google.oauth2 import service_account
from google.oauth2.service_account import Credentials
import google.auth
import google.auth.transport.requests
import requests as rqs
# here
import io
import textwrap
# DocAI
# pdfs
from pypdf import PdfReader, PdfWriter

logger = logging.getLogger()


class MatchingEngineVectorStore(VectorStore):
    """Vertex Matching Engine implementation of the vector store.

    While the embeddings are stored in the Matching Engine, the embedded
    documents will be stored in GCS.

    An existing Index and corresponding Endpoint are preconditions for
    using this module.

    See usage in docs/modules/indexes/vectorstores/examples/matchingengine.ipynb

    Note that this implementation is mostly meant for reading if you are
    planning to do a real time implementation. While reading is a real time
    operation, updating the index takes close to one hour."""

    def __init__(
        self
        , index: MatchingEngineIndex
        , endpoint: MatchingEngineIndexEndpoint
        , embedding: VertexEmbeddings()
        , gcs_client: storage.Client
        , index_client: aiplatform_v1.IndexServiceClient
        , index_endpoint_client: aiplatform_v1.IndexEndpointServiceClient
        , gcs_bucket_name: str
        , credentials: Credentials = None
        , project_num: str = '939655404703'
        , project_id: str = 'cpg-cdp'
        , region: str = 'us-central1'
        , k: int = 4
    ):
        """Vertex Matching Engine implementation of the vector store.

        While the embeddings are stored in the Matching Engine, the embedded
        documents will be stored in GCS.

        An existing Index and corresponding Endpoint are preconditions for
        using this module.

        See usage in
        docs/modules/indexes/vectorstores/examples/matchingengine.ipynb.

        Note that this implementation is mostly meant for reading if you are
        planning to do a real time implementation. While reading is a real time
        operation, updating the index takes close to one hour.

        Attributes:
            project_id: The GCS project id.
            index: The created index class. See
            ~:func:`MatchingEngine.from_components`.
            endpoint: The created endpoint class. See
            ~:func:`MatchingEngine.from_components`.
            embedding: A :class:`VertexEmbeddings` that will be used for
            embedding the text
            gcs_client: The Google Cloud Storage client.
            credentials (Optional): Created GCP credentials.
        """
        super().__init__()
        self._validate_google_libraries_installation()

        self.k = k
        self.project_id = project_id
        self.project_num = project_num
        self.region = region
        self.index = index
        self.endpoint = endpoint
        self.embedding = embedding
        self.gcs_client = gcs_client
        self.index_client = index_client
        self.index_endpoint_client = index_endpoint_client
        self.gcs_client = gcs_client
        self.credentials = credentials
        self.gcs_bucket_name = gcs_bucket_name

    def _validate_google_libraries_installation(self) -> None:
        """Validates that Google libraries that are needed are installed."""
        try:
            from google.cloud import aiplatform, storage  # noqa: F401
            from google.oauth2 import service_account  # noqa: F401
        except ImportError:
            raise ImportError(
                "You must run `pip install --upgrade "
                "google-cloud-aiplatform google-cloud-storage`"
                "to use the MatchingEngine Vectorstore."
            )

    def chunk_bq_table(
        self
        , bq_dataset_name: str
        , bq_table_name: str
        , query: str
        , page_content_cols: List[str]
        , metadata_cols: List[str]
        , chunk_size: int = 1000
        , chunk_overlap: int = 200
    ) -> List[Document]:
        """

        :param bq_dataset_name:
        :param bq_table_name:
        :param query:
        :param page_content_cols:
        :param metadata_cols:
        :param chunk_size:
        :return:
        """
        loader = BigQueryLoader(
            query=query
            , project=self.project_id
            , page_content_columns=page_content_cols
            , metadata_columns=metadata_cols
        )
        data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        doc_splits = text_splitter.split_documents(data)
        logging.info(f"# of chunked documents = {len(doc_splits)}")

        # TODO: test this
        # texts = [doc.page_content for doc in doc_splits]
        # metas = [doc.metadata for doc in doc_splits]
        # return texts, metas

        return [doc for doc in doc_splits]

    def chunk_unstructured_gcs_blob(
        self
        , blob_name: str
        , bucket_name: str
        , chunk_size: int = 1000
    ) -> List[Document]:
        """
        loads documents from GCS
        uses `UnstructuredFileLoader` which supports text, word.docx, ppt, html, pdfs, images
        """
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        logging.info(f"gcs_uri: {gcs_uri}")

        gcs_loader = GCSFileLoader(
            project_name=self.project_id
            , bucket=bucket_name
            , blob=blob_name
        )
        data = gcs_loader.load()
        for doc in data:
            doc.metadata['source'] = gcs_uri

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        doc_splits = text_splitter.split_documents(data)
        logging.info(f"# of documents = {len(doc_splits)}")

        return [doc for doc in doc_splits]
    
    def chunk_pdfs(
        self
        # , urls: List[str]
        , url: str
        , chunk_size: int = 1000
    ) -> List[Document]:
        """
        downloads source docs and breaks them into smaller chunks
        """
        loader = PyPDFLoader(url)
        data = loader.load()

        for doc in data:
            doc.metadata['source'] = url
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        doc_splits = text_splitter.split_documents(data)
        logging.info(f"# of documents = {len(doc_splits)}")
        
        # return [doc.page_content for doc in doc_splits]
        return [doc for doc in doc_splits]

    # TODO: May be deprecated?
    def extract_pdf_docai(
        self
        , file_path: str
        , docai_processor_id: str
        , mime_type: str
        , page_max: int = 15
    ) -> List[str]:
        """
        TODO: add original file uri to chunked metadata
        """
        pdf = PdfReader(open(f'{file_path}', "rb"))

        texts = self.process_pdf_pages(pdf, docai_processor_id, mime_type, page_max)
        logging.info(f"# of text chunks: {len(texts)}")

        return [doc.text for doc in texts]

    # TODO: Maybe deprecate?
    def process_pdf_pages(
        self
        , pdf
        , docai_processor_id: str
        , mime_type: str
        , page_max: int = 15
    ) -> List:
        # DocAI client
        client = documentai.DocumentProcessorServiceClient()
        docs = []
        page_count = 0
        for page in pdf.pages:
            page_count += 1
            buf = io.BytesIO()
            writer = PdfWriter()
            writer.add_page(page)
            writer.write(buf)
            process_request = {
                "name": f"projects/{self.project_num}/locations/us/processors/{docai_processor_id}",
                "raw_document": {
                    "content": buf.getvalue(),
                    "mime_type": mime_type,
                },
            }
            docs.append(client.process_document(request=process_request).document)
            # DocAI preprocessor page_limit=PAGE_MAX(15)
            if page_count == page_max:
                break

        return docs
        

    def chunk_powerpoint(
        self
        , local_file_path: str
        , chunk_size: int = 1000
    ) -> List[Document]:
        """
        downloads source docs and breaks them into smaller chunks
        """
        loader = UnstructuredPowerPointLoader(local_file_path)
        data = loader.load()

        logging.info(f"# of pages loaded (pre-chunking) = {len(data)}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        doc_splits = text_splitter.split_documents(data)
        logging.info(f"# of documents = {len(doc_splits)}")

        # return [doc.page_content for doc in doc_splits]
        return [doc for doc in doc_splits]

    def chunk_word_doc(
        self
        , local_file_path: str
        , chunk_size: int = 1000
    ) -> List[Document]:
        """
        downloads source docs and breaks them into smaller chunks
        """
        loader = UnstructuredWordDocumentLoader(local_file_path)
        data = loader.load()

        logging.info(f"# of pages loaded (pre-chunking) = {len(data)}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        doc_splits = text_splitter.split_documents(data)
        logging.info(f"# of documents = {len(doc_splits)}")

        # return [doc.page_content for doc in doc_splits]
        return [doc for doc in doc_splits]

    def chunk_youtube(
        self
        , youtube_id: str
        , youtube_prefix: str = 'https://www.youtube.com/watch?v='
        , add_video_info: bool = True
        , chunk_size: int = 1000
    ) -> List[Document]:
        # here
        loader = YoutubeLoader.from_youtube_url(
            f"{youtube_prefix}{youtube_id}"
            , add_video_info=True
        )
        data = loader.load()

        for doc in data:
            doc.metadata['source'] = f"{youtube_prefix}{youtube_id}"

        logging.info(f"# of pages loaded (pre-chunking) = {len(data)}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        doc_splits = text_splitter.split_documents(data)
        logging.info(f"# of documents = {len(doc_splits)}")

        return [doc for doc in doc_splits]

    def chunk_text(
        self
        , text: str
        , source: str
        , chunk_size: int = 1000
    ) -> List[Document]:
        """

        :param text:
        :param source:
        :param chunk_size:
        :return:
        """
        doc = Document()
        doc.page_content = text
        doc.metadata["source"] = source

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        doc_splits = text_splitter.split_documents(list(doc))
        logging.info(f"# of documents = {len(doc_splits)}")

        return [doc for doc in doc_splits]

    def summarize_docs(
        self
        , docs: Document
        , temperature: float = 0.0
        , max_output_tokens: int = 1000
        , top_p: float = 0.7
        , top_k: int = 40
    ) -> str:
        llm = VertexLLM(
            stop=None
            , temperature=temperature
            , max_output_tokens=max_output_tokens
            , top_p=top_p
            , top_k=top_k
        )

        chain = load_summarize_chain(
            llm
            , chain_type="map_reduce"
            , map_prompt=map_prompt
            , combine_prompt=combine_prompt
        )
        output_summary = chain.run(docs)

        wrapped_text = textwrap.fill(
            output_summary,
            width=200,
            break_long_words=False,
            replace_whitespace=False
        )

        return wrapped_text

    # TODO: Add or extend this to handle index creation in addition to upsert
    def add_texts(
        self
        , texts: Iterable[str]
        , metadatas: Optional[List[dict]] = None
        , **kwargs: dict[str, Any]
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        logging.info(f"# of texts = {len(texts)}")
        logging.info(f"# of metadatas = {len(metadatas)}")

        logger.debug("Embedding documents.")
        embeddings = self.embedding.embed_documents(list(texts))
        print(f"\nlen of embeddings: {len(embeddings)}")
        print(f"len of embeddings[0]: {len(embeddings[0])}")

        insert_datapoints_payload = []
        ids = []

        for idx, (embedding, text, metas) in enumerate(zip(embeddings, texts, metadatas)):
            id_ = uuid.uuid4()
            ids.append(id_)
            # embedding = vertex_embedding.embed_query(doc.page_content)
            self._upload_to_gcs(
                data=text
                , gcs_location=f"documents/{id_}"
                , metadata=metas
            )
            insert_datapoints_payload.append(
                aiplatform_v1.IndexDatapoint(
                    datapoint_id=str(id_)
                    , feature_vector=embedding
                )
            )
            if idx % 100 == 0:
                upsert_request = aiplatform_v1.UpsertDatapointsRequest(
                    index=self.index.name
                    , datapoints=insert_datapoints_payload
                )
                response = self.index_client.upsert_datapoints(request=upsert_request)
                insert_datapoints_payload = []

        if len(insert_datapoints_payload) > 0:
            upsert_request = aiplatform_v1.UpsertDatapointsRequest(
                index=self.index.name,
                datapoints=insert_datapoints_payload
            )
            response = self.index_client.upsert_datapoints(request=upsert_request)

        # logger.debug("Updated index with new configuration.")
        logger.info(f"Uploaded {len(ids)} documents to GCS.")

        return ids

    def _upload_to_gcs(
        self
        , data: str
        , gcs_location: str
        , metadata: Optional[dict]
    ) -> None:
        """Uploads data to gcs_location.

        Args:
            data: The data that will be stored.
            gcs_location: The location where the data will be stored.
        """
        bucket = self.gcs_client.get_bucket(self.gcs_bucket_name)
        blob = bucket.blob(gcs_location)
        if metadata:
            blob.metadata = metadata
        blob.upload_from_string(data)

    def get_matches(
            self
            , embeddings: List[float]
            , n_matches: int
            , index_endpoint: MatchingEngineIndexEndpoint
    ) -> str:
        '''
        get matches from matching engine given a vector query
        Uses public endpoint
        
        # TODO - add public endpoint capability

        '''
        
        ### SDK for indexes within VPC ###
        # index_endpoint.deployed_indexes[0].id
        deployed_index_id = self._get_index_id()
        
        response = index_endpoint.match(
            deployed_index_id=deployed_index_id
            , queries=[embeddings]
            , num_neighbors=n_matches
        )

        return response

    def get_matches_public_endpoint(
        self
        , embeddings: List[List[float]]
        , n_matches: int
        , endpoint_address: str
        , index_endpoint_id: str
    ) -> List[List[MatchNeighbor]]:
        '''
        get matches from matching engine given a vector query
        Uses public endpoint

        '''
        credentials, project = google.auth.default()
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        request_data = {
            "deployed_index_id": index_endpoint_id,
            'queries': [{
                'datapoint': {
                    "datapoint_id": f"{i}"
                    , "feature_vector": emb
                },
                'neighbor_count': n_matches}
                for i, emb in enumerate(embeddings)]
            }
        endpoint_json_data = json.dumps(request_data)

        logging.info(f"Data payload: {endpoint_json_data}")
        rpc_address = f'https://{endpoint_address}/v1beta1/projects/{self.project_num}/locations/{self.region}/indexEndpoints/{index_endpoint_id}:findNeighbors'
        logging.info(f"RPC Address for public endpoint request: {rpc_address}")

        header = {'Authorization': 'Bearer ' + credentials.token}

        response: Response = rqs.post(
            url=rpc_address
            , data=endpoint_json_data
            , headers=header
        )
        response.json()
        logging.info(json.dumps(response.__dict__))
        logging.info(f"response text: {response.text}, response content: {response.content}")
        final_list = []
        for query in dict(response.json())["nearestNeighbors"]:
            tmp_neighbors = [MatchNeighbor(
                id=dp["datapoint"]["datapointId"]
                , distance=dp["distance"]
            ) for dp in query["neighbors"]]
            final_list.append(tmp_neighbors)

        return final_list

    def similarity_search(
        self
        , query: str
        , k: int = None
        , **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: The string that will be used to search for similar documents.
            k: The amount of neighbors that will be retrieved.

        Returns:
            A list of k matching documents.
        """

        k = k if k is not None else self.k
        logger.info(f"Embedding query {query}.")
        embedding_query = self.embedding.embed_documents([query])
        deployed_index_id = self._get_index_id()
        logger.info(f"Deployed Index ID = {deployed_index_id}")

        if (
            self.endpoint.public_endpoint_domain_name != ''
            # or self.endpoint.public_endpoint_domain_name is not None
        ):
            response = self.get_matches_public_endpoint(
                embeddings=embedding_query
                , n_matches=k
                , endpoint_address=self.endpoint.public_endpoint_domain_name
                , index_endpoint_id=self.endpoint.name
            )
        else:
            response = self.endpoint.match(
                deployed_index_id=self._get_index_id(),
                queries=embedding_query,
                num_neighbors=k,
            )
        
        results = []
        
        for match in response[0]:
            page_content = self._download_from_gcs(f"documents/{match.id}")
            metadata = self._get_gcs_blob_metadata(f"documents/{match.id}")
            if metadata:
                results.append(
                    Document(
                        page_content=page_content
                        , metadata=metadata
                        # , metadata={'source': metadata['source']}
                    )
                )
            else:
                results.append(
                    Document(
                        page_content=page_content
                        , metadata={
                            "match_id": f"{match.id}"
                        }
                    )
                )

        return results

    def _get_index_id(self) -> str:
        """Gets the correct index id for the endpoint.

        Returns:
            The index id if found (which should be found) or throws
            ValueError otherwise.
        """
        for index in self.endpoint.deployed_indexes:
            if index.index == self.index.name:
                return index.id

        raise ValueError(
            f"No index with id {self.index.name} "
            f"deployed on enpoint "
            f"{self.endpoint.display_name}."
        )

    def _download_from_gcs(self, gcs_location: str) -> str:
        """Downloads from GCS in text format.

        Args:
            gcs_location: The location where the file is located.

        Returns:
            The string contents of the file.
        """
        bucket = self.gcs_client.get_bucket(self.gcs_bucket_name)
        try:
            blob = bucket.blob(gcs_location)
            return blob.download_as_string()
        except Exception as e:
            return ''
        
    def _get_gcs_blob_metadata(self, gcs_location: str) -> dict:
        """Downloads GCS blob metadata

        Args:
            gcs_location: The location where the file is located.

        Returns:
            dictionary of metadata
        """
        bucket = self.gcs_client.get_bucket(self.gcs_bucket_name)
        try:
            blob = bucket.get_blob(gcs_location)
            return blob.metadata
        except Exception as e:
            return {'source': ''}

    @classmethod
    def from_texts(
        cls: Type["MatchingEngineVectorStore"],
        texts: List[str],
        embedding: VertexEmbeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "MatchingEngineVectorStore":
        """Use from components instead."""
        raise NotImplementedError(
            "This method is not implemented. Instead, you should initialize the class"
            " with `MatchingEngine.from_components(...)` and then call "
            "`from_texts`"
        )

    @classmethod
    def from_documents(
        cls: Type["MatchingEngineVectorStore"],
        documents: List[str],
        embedding: VertexEmbeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "MatchingEngineVectorStore":
        """Use from components instead."""
        raise NotImplementedError(
            "This method is not implemented. Instead, you should initialize the class"
            " with `MatchingEngine.from_components(...)` and then call "
            "`from_documents`"
        )

    # 'project_id', 'region', 'index_name', and 'vpc_network_name'
    @classmethod
    def from_components(
        cls: Type["MatchingEngineVectorStore"]
        , project_id: str
        , region: str
        , gcs_bucket_name: str
        , index_id: str
        , endpoint_id: str
        , credentials_path: Optional[str] = None
        , embedding: Optional[VertexEmbeddings] = None
        , k: int = 4
    ) -> "MatchingEngineVectorStore":
        """Takes the object creation out of the constructor.

        Args:
            project_id: The GCP project id.
            region: The default location making the API calls. It must have
            the same location as the GCS bucket and must be regional.
            gcs_bucket_name: The location where the vectors will be stored in
            order for the index to be created.
            index_id: The id of the created index.
            endpoint_id: The id of the created endpoint.
            credentials_path: (Optional) The path of the Google credentials on
            the local file system.
            embedding: The :class:`VertexEmbeddings` that will be used for
            embedding the texts.

        Returns:
            A configured MatchingEngine with the texts added to the index.
        """
        gcs_bucket_name = cls._validate_gcs_bucket(gcs_bucket_name)

        # Set credentials
        if credentials_path:
            credentials = cls._create_credentials_from_file(credentials_path)
        else:
            credentials, _ = google.auth.default()
            request = google.auth.transport.requests.Request()
            credentials.refresh(request)

        index = cls._create_index_by_id(
            index_id=index_id
            , project_id=project_id
            , region=region
            , credentials=credentials
        )
        endpoint = cls._create_endpoint_by_id(
            endpoint_id=endpoint_id
            , project_id=project_id
            , region=region
            , credentials=credentials
        )

        gcs_client = cls._get_gcs_client(
            credentials=credentials
            , project_id=project_id
        )
        index_client = cls._get_index_client(
            project_id=project_id
            , region=region
            , credentials=credentials
        )
        index_endpoint_client = cls._get_index_endpoint_client(
            project_id=project_id
            , region=region
            , credentials=credentials
        )
        cls._init_aiplatform(
            project_id=project_id
            , region=region
            , gcs_bucket_name=gcs_bucket_name
            , credentials=credentials
        )

        return cls(
            project_id=project_id,
            region=region,
            index=index,
            endpoint=endpoint,
            embedding=embedding,#or cls._get_default_embeddings(),
            gcs_client=gcs_client,
            index_client=index_client,
            index_endpoint_client=index_endpoint_client,
            credentials=credentials,
            gcs_bucket_name=gcs_bucket_name,
            k=k
        )

    @classmethod
    def _validate_gcs_bucket(cls, gcs_bucket_name: str) -> str:
        """Validates the gcs_bucket_name as a bucket name.

        Args:
              gcs_bucket_name: The received bucket uri.

        Returns:
              A valid gcs_bucket_name or throws ValueError if full path is
              provided.
        """
        gcs_bucket_name = gcs_bucket_name.replace("gs://", "")
        if "/" in gcs_bucket_name:
            raise ValueError(
                f"The argument gcs_bucket_name should only be "
                f"the bucket name. Received {gcs_bucket_name}"
            )
        return gcs_bucket_name

    @classmethod
    def _create_credentials_from_file(
        cls, json_credentials_path: Optional[str]
    ) -> Optional[Credentials]:
        """Creates credentials for GCP.

        Args:
             json_credentials_path: The path on the file system where the
             credentials are stored.

         Returns:
             An optional of Credentials or None, in which case the default
             will be used.
        """

        credentials = None
        if json_credentials_path is not None:
            credentials = service_account.Credentials.from_service_account_file(
                json_credentials_path
            )

        return credentials

    @classmethod
    def _create_index_by_id(
        cls, index_id: str, project_id: str, region: str, credentials: "Credentials"
    ) -> MatchingEngineIndex:
        """Creates a MatchingEngineIndex object by id.

        Args:
            index_id: The created index id.

        Returns:
            A configured MatchingEngineIndex.
        """
        logger.debug(f"Creating matching engine index with id {index_id}.")
        index_client = cls._get_index_client(
            project_id=project_id
            , region=region
            , credentials=credentials
        )
        full_index_uri = f"projects/{project_id}/locations/{region}/indexes/{index_id}"
        request = aiplatform_v1.GetIndexRequest(name=full_index_uri)
        return index_client.get_index(request=request)

    @classmethod
    def _create_endpoint_by_id(
        cls, endpoint_id: str, project_id: str, region: str, credentials: "Credentials"
    ) -> MatchingEngineIndexEndpoint:
        """Creates a MatchingEngineIndexEndpoint object by id.

        Args:
            endpoint_id: The created endpoint id.

        Returns:
            A configured MatchingEngineIndexEndpoint.
            :param project_id:
            :param region:
            :param credentials:
        """
        logger.debug(f"Creating endpoint with id {endpoint_id}.")
        return aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_id,
            project=project_id,
            location=region,
            credentials=credentials,
        )

    @classmethod
    def _get_gcs_client(
        cls, credentials: "Credentials", project_id: str
    ) -> "storage.Client":
        """Lazily creates a GCS client.

        Returns:
            A configured GCS client.
        """
        return storage.Client(credentials=credentials, project=project_id)

    @classmethod
    def _get_index_client(
        cls, project_id: str, region: str, credentials: "Credentials"
    ) -> "aiplatform_v1.IndexServiceClient":
        """Lazily creates a Matching Engine Index client.

        Returns:
            A configured Matching Engine Index client.
        """
        #PARENT = f"projects/{project_id}/locations/{region}"
        ENDPOINT = f"{region}-aiplatform.googleapis.com"
        return aiplatform_v1.IndexServiceClient(
            client_options=dict(api_endpoint=ENDPOINT),
            credentials=credentials
        )

    @classmethod
    def _get_index_endpoint_client(
        cls, project_id: str, region: str, credentials: "Credentials"
    ) -> "aiplatform_v1.IndexEndpointServiceClient":
        """Lazily creates a Matching Engine Index Endpoint client.

        Returns:
            A configured Matching Engine Index Endpoint client.
        """
        #PARENT = f"projects/{project_id}/locations/{region}"
        ENDPOINT = f"{region}-aiplatform.googleapis.com"
        return aiplatform_v1.IndexEndpointServiceClient(
            client_options=dict(api_endpoint=ENDPOINT),
            credentials=credentials
        )

    @classmethod
    def _init_aiplatform(
        cls,
        project_id: str,
        region: str,
        gcs_bucket_name: str,
        credentials: "Credentials",
    ) -> None:
        """Configures the aiplatform library.

        Args:
            project_id: The GCP project id.
            region: The default location making the API calls. It must have
            the same location as the GCS bucket and must be regional.
            gcs_bucket_name: GCS staging location.
            credentials: The GCS Credentials object.
        """
        logger.debug(
            f"Initializing AI Platform for project {project_id} on "
            f"{region} and for {gcs_bucket_name}."
        )
        aiplatform.init(
            project=project_id,
            location=region,
            credentials=credentials,
        )

    @classmethod
    def _get_default_embeddings(cls) -> TensorflowHubEmbeddings:
        """This function returns the default embedding."""
        return TensorflowHubEmbeddings()
