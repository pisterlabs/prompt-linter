from unittest.mock import Mock, patch
import pytest
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.schema import Document

from src.service.chatbot.loaders.s3_file_loader import S3FileLoader


def mock_s3_connector():
    # Create a mock S3Connector
    mock_s3_connector = Mock()
    mock_s3_connector.download_object.return_value = None  # Simulate download success
    return mock_s3_connector


def mock_loader():
    loader = Mock()
    loader.load.return_value = PyPDFLoader("./resources/sample.pdf").load()
    return loader


def mock_loader_txt():
    loader = Mock()
    loader.load.return_value = UnstructuredFileLoader("./resources/sample.txt").load()
    return loader


@patch("src.service.chatbot.loaders.file_loader_factory.FileLoaderFactory.get_loader")
@patch("src.api.s3.S3Connector.__new__")
def test_load_with_pdf_file(mock_1, mock_2):
    mock_1.return_value = mock_s3_connector()
    mock_2.return_value = mock_loader_pdf()
    # Test loading a PDF file from S3
    object_id = "./resources/sample.pdf"

    s3_file_loader = S3FileLoader(object_id)
    documents = s3_file_loader.load()
    assert len(documents) == 6
    assert isinstance(documents[0], Document)
