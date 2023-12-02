import boto3
import logging
from dotenv import load_dotenv
from botocore.exceptions import ClientError
import tempfile
from langchain.schema.document import Document
from typing import Tuple, List, Any
from langchain.document_loaders import TextLoader
import os


# Load environment variables from the .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)


def process_audio(s3_client: boto3.client, bucket_name: str, file_key: str, openai_client: Any) -> Tuple[List[Document], str]:
    """
    Download a audio file from an S3 bucket, process it to extract text, and return the extracted text and temporary file path.

    Parameters:
    s3_client (boto3.client): The S3 client used to interact with Amazon S3.
    bucket_name (str): The name of the S3 bucket where the audio file is stored.
    file_key (str): The key of the PDF file in the S3 bucket.
    openai_client (Any): The OpenAI client used to interact with OpenAI.

    Returns:
    tuple: A tuple containing the extracted text data and the temporary file path of the downloaded PDF.
    """
    try:
        logging.info(f"Downloading audio file from S3: {file_key}")

        # Download the audio file from S3
        s3_object = s3_client.get_object(
            Bucket=os.environ.get('YOUR_BUCKET_NAME'), Key=file_key)
        audio_file_bytes = s3_object['Body'].read()

        # Get the file extension
        _, file_extension = os.path.splitext(file_key)

        # Check if the file is an MP3 or M4A
        if file_extension.lower() in (".mp3", ".m4a"):
            # Create a temporary file to store the downloaded audio file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_audio_file:
                temp_audio_file.write(audio_file_bytes)
                temp_audio_path = temp_audio_file.name

            logging.info(f"Processing audio: {file_key}")

            audio_file = open(temp_audio_path, 'rb')
            result = openai_client.audio.transcriptions.create(
                model='whisper-1', file=audio_file, response_format='text')
            audio_file.close()

            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_audio_file:
                temp_audio_file.write(result)
                temp_audiot_path = temp_audio_file.name

            # Load data from the temporary file
            loader = TextLoader(temp_audiot_path)
            data = loader.load()

            logging.info(f"Audio processed successfully: {file_key}")

            # Return both 'data' and 'temp_audio_path'
            return data, temp_audiot_path
        else:
            raise TypeError("Error")

    except ClientError as e:
        logging.error(f"S3 client error: {e}")
        raise Exception(f"S3 client error: {e}")

    except Exception as e:
        logging.error(f"Error processing the audio file: {e}")
        raise Exception(f"Error processing the audio file: {e}")

    # Delete the temporary file
    finally:
        if 'temp_audio_path' in locals() and 'temp_audiot_path' in locals():
            # Delete the temporary file
            os.remove(temp_audiot_path)
