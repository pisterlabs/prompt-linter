# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import json
import boto3
from botocore.exceptions import ClientError
import langchain
from langchain.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector

from langchain.indexes import SQLRecordManager
from langchain.indexes import index
import sqlalchemy

from utils import bedrock

def lambda_handler(event, context):
    # Get content of uploaded object
    s3 = boto3.client('s3')
    s3_details = event["Records"][0]["s3"]
    response = s3.get_object(Bucket=s3_details["bucket"]["name"], Key=s3_details["object"]["key"])
    content = response['Body'].read().decode('utf-8')

    # Set up client for Amazon Bedrock
    boto3_bedrock = bedrock.get_bedrock_client(region="us-east-1")
    br_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=boto3_bedrock)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = splitter.create_documents([content], metadatas=[{"source": s3_details["object"]["key"]}]);
    print(f"Number of documents after split and chunking = {len(docs)}")

    # Retrieve database credentials from AWS Secrets Manager
    db_credential = get_db_credential()
    pgvector_connection_string = PGVector.connection_string_from_db_params(
        driver="psycopg2",
        host=db_credential["host"],
        port=int(db_credential["port"]),
        database=db_credential["username"],
        user=db_credential["username"],
        password=db_credential["password"],
    )

    # Record Manager is used to load and keep in sync documents from any source into a vector store
    # https://blog.langchain.dev/syncing-data-sources-to-vector-stores/
    collection_name = "knowledge_base"
    namespace = f"pgvector/{collection_name}"
    record_manager = SQLRecordManager(
        namespace, engine=sqlalchemy.create_engine("postgresql+psycopg2://postgres:" + db_credential["password"] + "@" + db_credential["host"] + "/postgres")
    )
    record_manager.create_schema()

    # Create vector store
    vectorstore_pgvector_aws = PGVector(pgvector_connection_string, br_embeddings, collection_name=collection_name)

    # Create embeddings and store in vector store
    index(
        docs_source=docs,
        record_manager=record_manager,
        vector_store=vectorstore_pgvector_aws,
        cleanup="incremental",
        source_id_key="source"
    )

    # Performing a query for testing
    print("Performing a query for testing")
    print("-" * 35)
    query = "How do the new features of AWS Health help me?"
    docs_with_score = vectorstore_pgvector_aws.similarity_search_with_score(query)
    for doc, score in docs_with_score:
        print("Score: ", score)
        print(doc.page_content)
        print("-" * 35)

    print("boto3: " + boto3.__version__ + ", langchain: " + langchain.__version__)

def get_db_credential():
    secret_name = "text-embeddings-pipeline-vector-store"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager'
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    return json.loads(get_secret_value_response['SecretString'])