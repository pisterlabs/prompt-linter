import boto3
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import os
from pathlib import Path
import random
import re
import requests
from requests.auth import HTTPBasicAuth
import string
import tiktoken
from tqdm import tqdm
import traceback
from urllib.parse import unquote_plus

logger = logging.getLogger(__name__)
if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)

s3_client = boto3.client('s3')
sagemaker_runtime_client = boto3.client('sagemaker-runtime')

es_username = os.getenv("ES_USERNAME", default=None)
es_password = os.getenv("ES_PASSWORD", default=None)
es_url = os.getenv("ES_URL", default=None)
es_index_name = os.getenv("ES_INDEX_NAME", default=None)
sagemaker_endpoint = os.getenv("SAGEMAKER_ENDPOINT", default=None)

encoding = tiktoken.get_encoding('cl100k_base')
CHUNK_SIZE = 768
CHUNK_SIZE_MIN = 20
output_file_path = "/tmp/docs"

def create_index(url):
    try:
        print("Creating Index")
        mapping = {
            'settings': {
                'index': {
                    'knn': True  # Enable k-NN search for this index
                }
            },
            'mappings': {
                'properties': {
                    'embedding': {  # k-NN vector field
                        'type': 'knn_vector',
                        'dimension': 4096,  # Dimension of the vector
                        'similarity': 'l2_norm'
                    },
                    'file_name': {
                        'type': 'text'
                    },
                    'page': {
                        'type': 'text'
                    },
                    'passage': {
                        'type': 'text'
                    }
                }
            }
        }

        response = requests.put(url, auth=HTTPBasicAuth(es_username, es_password), json=mapping)
        print(f'Index created: {response.text}')
    except Exception as e:
        stacktrace = traceback.format_exc()
        print("{}".format(stacktrace))

        raise e

def delete_index(url):
    try:
        response = requests.head(url, auth=HTTPBasicAuth(es_username, es_password))

        if response.status_code != 404:
            print('Index already exists! Deleting...')
            response = requests.delete(url, auth=HTTPBasicAuth(es_username, es_password))

            print(response.text)
    except Exception as e:
        stacktrace = traceback.format_exc()
        print("{}".format(stacktrace))

        raise e

def doc_iterator(dir_path: str):
    for root, _, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            page = filename.split(".")[0].split("_")[-1]
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    file_contents = file.read()
                    yield filename, page, file_contents

def get_chunks(file_name, file_path):
    try:
        print("Get file chunks")
        chunks = []
        total_passages = 0

        for doc_name, page, doc in tqdm(doc_iterator(file_path)):
            n_passages = 0

            doc = re.sub(r"(\w)-\n(\w)", r"\1\2", doc)
            doc = re.sub(r"(?<!\n)\n(?!\n)", " ", doc)
            doc = re.sub(r"\n{2,}", "\n", doc)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                chunk_overlap=200,
            )

            tmp_chunks = text_splitter.split_text(doc)

            for i, chunk in enumerate(tmp_chunks):
                chunks.append({
                    "file_name": file_name,
                    "page": page,
                    "passage": chunk
                })
                n_passages += 1
                total_passages += 1

            logger.info(f'Document segmented into {n_passages} passages')

        logger.info(f'Total passages to index: {total_passages}')

        return chunks
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e

def get_txt_file(bucket_name, object_key, file_path):
    try:
        logger.info("Get txt file")
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        data = response['Body'].read().decode("utf-8")

        logger.info(data)

        f = open("{}/output_1.txt".format(file_path), "a")
        f.write(data)
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e

def index_documents(url, chunks):
    try:
        logger.info("Indexing documents")

        i = 1
        for chunk in chunks:
            payload = {'text_inputs': [chunk["passage"]]}
            payload = json.dumps(payload).encode('utf-8')

            response = sagemaker_runtime_client.invoke_endpoint(EndpointName=sagemaker_endpoint,
                                                         ContentType='application/json',
                                                         Body=payload)

            model_predictions = json.loads(response['Body'].read())
            embedding = model_predictions['embedding'][0]

            document = {
                'embedding': embedding,
                'file_name': chunk["file_name"],
                'page': chunk["page"],
                "passage": chunk["passage"]
            }

            response = requests.post(f'{url}/_doc/{i}', auth=HTTPBasicAuth(es_username, es_password), json=document)
            i += 1

            logger.info(response.text)

            if response.status_code not in [200, 201]:
                logger.info(response.status_code)
                logger.info(response.text)
                break

    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e

def lambda_handler(event, context):
    try:
        logger.info(event)

        if "Records" in event:
            event_type = event["Records"][0]["eventName"]

            if event_type and event_type != "ObjectRemoved:Delete":
                bucket_name = event["Records"][0]["s3"]["bucket"]["name"]
                object_key = event["Records"][0]["s3"]["object"]["key"]
                object_key = unquote_plus(object_key)
                file_name = object_key.split("/")[-1]
                job_id = ''.join(random.choices(string.ascii_letters, k=15))

                logger.info("Bucket {}".format(bucket_name))
                logger.info("Object {}".format(object_key))

                path = Path(os.path.join(output_file_path, job_id))
                path.mkdir(parents=True, exist_ok=True)

                get_txt_file(bucket_name, object_key, os.path.join(output_file_path, job_id))

                chunks = get_chunks(file_name, os.path.join(output_file_path, job_id))

                if len(object_key.split("/")) == 5:
                    index_name = object_key.split("/")[3]
                    new_es_url = es_url + "/" + es_index_name + "-" + index_name
                else:
                    new_es_url = es_url + "/" + es_index_name

                delete_index(new_es_url)

                create_index(new_es_url)

                index_documents(new_es_url, chunks)

        return {
            'statusCode': 200,
            'body': json.dumps('Indexing finished')
        }
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e
