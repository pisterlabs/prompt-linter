import boto3
from langchain.document_loaders import PyPDFLoader
import logging
from dotenv import load_dotenv
import os
from botocore.exceptions import ClientError
import tempfile

# Load environment variables from the .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)


def process_pdf(s3_client: boto3.client, bucket_name: str, file_key: str) -> tuple:
    """
    Download a PDF file from an S3 bucket, process it to extract text, and return the extracted text and temporary file path.

    Parameters:
    s3_client (boto3.client): The S3 client used to interact with Amazon S3.
    bucket_name (str): The name of the S3 bucket where the PDF file is stored.
    file_key (str): The key of the PDF file in the S3 bucket.

    Returns:
    tuple: A tuple containing the extracted text data and the temporary file path of the downloaded PDF.
    """
    try:
        logging.info(f"Downloading PDF from S3: {file_key}")

        # Download the PDF from S3
        s3_object = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        pdf_file_bytes = s3_object['Body'].read()

        # Create a temporary file to store the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
            temp_pdf_file.write(pdf_file_bytes)
            temp_pdf_path = temp_pdf_file.name

        logging.info(f"Processing PDF: {file_key}")

        # Process the PDF
        loader = PyPDFLoader(temp_pdf_path)
        data = loader.load()

        logging.info(f"PDF processed successfully: {file_key}")

        # Return both 'data' and 'temp_pdf_path'
        return data, temp_pdf_path

    except ClientError as e:
        logging.error(f"S3 client error: {e}")
        raise Exception(f"S3 client error: {e}")

    except Exception as e:
        # You can log or re-raise the exception as needed
        logging.error(f"Error processing the PDF: {e}")
        raise Exception(f"Error processing the PDF: {e}")
