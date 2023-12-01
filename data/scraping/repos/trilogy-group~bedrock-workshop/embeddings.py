import boto3
from langchain.storage import LocalFileStore
from langchain.embeddings import BedrockEmbeddings, CacheBackedEmbeddings

aws_region = 'us-east-1'
bedrock_client = boto3.client("bedrock-runtime", aws_region)

