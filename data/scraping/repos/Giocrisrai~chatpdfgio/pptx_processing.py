import boto3
from langchain.document_loaders import UnstructuredPowerPointLoader
import logging
from botocore.exceptions import ClientError
import tempfile
from langchain.schema.document import Document
from typing import Tuple, List


def process_pptx(s3_client: boto3.client, bucket_name: str, file_key: str) -> Tuple[List[Document], str]:
    """
    Download a .pptx file from an S3 bucket, process it to extract text, and return the extracted text and temporary file path.

    Parameters:
    s3_client (boto3.client): The S3 client used to interact with Amazon S3.
    bucket_name (str): The name of the S3 bucket where the PPTX file is stored.
    file_key (str): The key of the PPTX file in the S3 bucket.

    Returns:
    tuple: A tuple containing the extracted text data and the temporary file path of the downloaded PPTX.
    """
    try:
        logging.info(f"Downloading PPTX from S3: {file_key}")

        # Download the PPTX from S3
        s3_object = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        pptx_file_bytes = s3_object['Body'].read()

        # Create a temporary file to store the PPTX
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pptx') as temp_pptx_file:
            temp_pptx_file.write(pptx_file_bytes)
            temp_pptx_path = temp_pptx_file.name

        logging.info(f"Processing PPTX: {file_key}")

        # Process the PPTX
        loader = UnstructuredPowerPointLoader(temp_pptx_path)
        data = loader.load()

        logging.info(f"PPTX processed successfully: {file_key}")

        # Return both 'data' and 'temp_pptx_path'
        return data, temp_pptx_path

    except ClientError as e:
        logging.error(f"S3 client error: {e}")
        raise Exception(f"S3 client error: {e}")

    except Exception as e:
        # You can log or re-raise the exception as needed
        logging.error(f"Error processing the PPTX: {e}")
        raise Exception(f"Error processing the PPTX: {e}")
