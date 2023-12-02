import boto3
from langchain.document_loaders import UnstructuredWordDocumentLoader
import logging
from botocore.exceptions import ClientError
import tempfile
from langchain.schema.document import Document
from typing import Tuple, List


def process_docx(s3_client: boto3.client, bucket_name: str, file_key: str) -> Tuple[List[Document], str]:
    """
    Download a .docx file from an S3 bucket, process it to extract text, and return the extracted text and temporary file path.

    Parameters:
    s3_client (boto3.client): The S3 client used to interact with Amazon S3.
    bucket_name (str): The name of the S3 bucket where the DOCX file is stored.
    file_key (str): The key of the DOCX file in the S3 bucket.

    Returns:
    tuple: A tuple containing the extracted text data and the temporary file path of the downloaded DOCX.
    """
    try:
        logging.info(f"Downloading DOCX from S3: {file_key}")

        # Download the DOCX from S3
        s3_object = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        docx_file_bytes = s3_object['Body'].read()

        # Create a temporary file to store the DOCX
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_docx_file:
            temp_docx_file.write(docx_file_bytes)
            temp_docx_path = temp_docx_file.name

        logging.info(f"Processing DOCX: {file_key}")

        # Process the DOCX
        loader = UnstructuredWordDocumentLoader(temp_docx_path)
        data = loader.load()

        logging.info(f"DOCX processed successfully: {file_key}")

        # Return both 'data' and 'temp_docx_path'
        return data, temp_docx_path

    except ClientError as e:
        logging.error(f"S3 client error: {e}")
        raise Exception(f"S3 client error: {e}")

    except Exception as e:
        # You can log or re-raise the exception as needed
        logging.error(f"Error processing the DOCX: {e}")
        raise Exception(f"Error processing the DOCX: {e}")
