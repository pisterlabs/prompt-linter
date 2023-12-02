import io
import os
import re
from typing import Tuple
import warlock
from dto_config_handler.output import ConfigDTO
from dto_events_handler.shared import StatusDTO
from pylog.log import setup_logging
from PyPDF2 import PdfReader
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pyminio.client import minio_client, MinioClient

logger = setup_logging(__name__)

class Job:
    """
    Handles the processing of document data.

    Args:
        config (ConfigDTO): The configuration data.
        input_data (type[warlock.model.Model]): The input data for the job.
        embeddings: Embeddings for document processing.
        dimension: Dimension for embeddings.

    Attributes:
        _config (ConfigDTO): The configuration data.
        _source (str): The source identifier.
        _context (str): The context environment.
        _input_data (type[warlock.model.Model]): The input data for the job.
        _embeddings: Embeddings for document processing.
        _dimension: Dimension for embeddings.
        _partition (str): The partition identifier.
        _target_endpoint (str): The document's target endpoint.
        _neo4j_url (str): The URL for Neo4j database.
        _neo4j_username (str): The username for Neo4j database.
        _neo4j_password (str): The password for Neo4j database.

    Methods:
        __init__(self, config: ConfigDTO, input_data: type[warlock.model.Model], embeddings, dimension) -> None:
            Initializes the Job instance.

        _get_bucket_name(self, layer: str) -> str:
            Generates the bucket name for Minio storage.

        _get_status(self) -> StatusDTO:
            Gets the success status.

        _get_file_path(self):
            Extracts the file path from the target endpoint.

        get_pdf_from_bucket(self, minio: MinioClient) -> PdfReader:
            Downloads and returns the PDF file from Minio.

        _convert_document_to_text(self, pdf_reader: PdfReader) -> str:
            Converts the PDF document to text.

        split_document(self, pdf_reader: PdfReader):
            Splits the document into chunks using langchain_textsplitter.

        get_neo4j_credentials(self):
            Retrieves the Neo4j database credentials.

        store_embeddings(self, chunks):
            Stores the document chunks in the Neo4j database.

        run(self) -> Tuple[dict, StatusDTO]:
            Runs the document processing job.

    """
    def __init__(self, config: ConfigDTO, input_data: type[warlock.model.Model], embeddings, dimension) -> None:
        """
        Initializes the Job instance.

        Args:
            config (ConfigDTO): The configuration data.
            input_data (type[warlock.model.Model]): The input data for the job.
            embeddings: Embeddings for document processing.
            dimension: Dimension for embeddings.

        Returns:
            None
        """
        self._config = config
        self._source = config.source
        self._context = config.context
        self._input_data = input_data
        self._embeddings = embeddings
        self._dimension = dimension
        self._partition = input_data.partition
        self._target_endpoint = input_data.documentUri
        self._neo4j_url, self._neo4j_username, self._neo4j_password = self.get_neo4j_credentials()

    def _get_bucket_name(self, layer: str) -> str:
        """
        Generates the bucket name for Minio storage.

        Args:
            layer (str): The layer of the bucket.

        Returns:
            str: The bucket name.
        """
        return "{layer}-{context}-source-{source}".format(
            layer=layer,
            context=self._context,
            source=self._source,
        )

    def _get_status(self) -> StatusDTO:
        """
        Gets the success status.

        Returns:
            StatusDTO: The success status.
        """
        return StatusDTO(
            code=200,
            detail="Success",
        )

    def _get_file_path(self):
        """
        Extracts the file path from the target endpoint.

        Returns:
            None
        """
        match = re.search(f"{self._partition}.*", self._target_endpoint)
        if match:
            return match.group()
        else:
            logger.warning("Year not found in onclick attribute")

    def get_pdf_from_bucket(self, minio: MinioClient) -> PdfReader:
        """
        Downloads and returns the PDF file from Minio.

        Args:
            minio (MinioClient): The Minio client.

        Returns:
            PdfReader: The PDF file reader.
        """
        logger.info(f"endpoint: {self._target_endpoint}")
        file_bytes = minio.download_file_as_bytes(self._get_bucket_name(layer="landing"), self._get_file_path())
        # TODO: AttributeError: 'bytes' object has no attribute 'seek'
        return PdfReader(io.BytesIO(file_bytes))

    def _convert_document_to_text(self, pdf_reader: PdfReader) -> str:
        """
        Converts the PDF document to text.

        Args:
            pdf_reader (PdfReader): The PDF file reader.

        Returns:
            str: The text extracted from the document.
        """
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def split_document(self, pdf_reader: PdfReader):
        """
        Splits the document into chunks using langchain_textsplitter.

        Args:
            pdf_reader (PdfReader): The PDF file reader.

        Returns:
            None
        """
        text = self._convert_document_to_text(pdf_reader)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

        chunks = text_splitter.split_text(text=text)
        return chunks

    def get_neo4j_credentials(self):
        """
        Retrieves the Neo4j database credentials.

        Returns:
            Tuple[str, str, str]: The Neo4j database URL, username, and password.
        """
        url = os.getenv("NEO4J_URL")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        return url, username, password

    def store_embeddings(self, chunks):
        """
        Stores the document chunks in the Neo4j database.

        Args:
            chunks: The document chunks.

        Returns:
            None
        """
        vectorstore = Neo4jVector.from_texts(
            chunks,
            url=self._neo4j_url,
            username=self._neo4j_username,
            password=self._neo4j_password,
            embedding=self._embeddings,
            index_name="pdf_enbeddings",
            node_label="PdfEnbeddingsChunk",
            pre_delete_collection=False,
        )

    def run(self) -> Tuple[dict, StatusDTO]:
        """
        Runs the document processing job.

        Returns:
            Tuple[dict, StatusDTO]: A tuple containing job result and status.
        """
        logger.info(f"Job triggered with input: {self._input_data}")
        minio = minio_client()
        pdf_reader = self.get_pdf_from_bucket(minio)
        document_chunks = self.split_document(pdf_reader)
        self.store_embeddings(document_chunks)
        result = {"documentUri": "", "partition": self._partition}
        logger.info(f"Job result: {result}")
        return result, self._get_status()
