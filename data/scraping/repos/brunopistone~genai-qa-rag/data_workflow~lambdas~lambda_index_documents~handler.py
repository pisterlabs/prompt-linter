import boto3
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import os
from pathlib import Path
import re
import requests
from requests.auth import HTTPBasicAuth
from textractcaller.t_call import get_full_json, Textract_API
from tqdm import tqdm
from trp import Document
import traceback

logger = logging.getLogger(__name__)
if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)

s3_client = boto3.client('s3')
sagemaker_runtime_client = boto3.client('sagemaker-runtime')
textract_client = boto3.client("textract")

es_username = os.getenv("ES_USERNAME", default=None)
es_password = os.getenv("ES_PASSWORD", default=None)
es_url = os.getenv("ES_URL", default=None)
es_index_name = os.getenv("ES_INDEX_NAME", default=None)
sagemaker_endpoint = os.getenv("SAGEMAKER_ENDPOINT", default=None)

CHUNK_SIZE = 768
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
                        'similarity': 'cosine'
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

def extract_blocks(job_id, job_status, file_path):
    if job_status == "SUCCEEDED":
        blocks = get_full_json(
            job_id,
            textract_api=Textract_API.ANALYZE,
            boto3_textract_client=textract_client
        )

        write_blocks(blocks, file_path)

def get_chunks(file_name, object_key, file_path):
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

def write_blocks(textract_resp, file_path):
    try:
        doc = Document(textract_resp)

        page_number = 1
        for page in doc.pages:
            print("Page ", page_number)

            text = ""

            for line in page.lines:
                text = text + " " + line.text

            # Print tables
            for table in page.tables:
                text = text + "\n\n"
                for r, row in enumerate(table.rows):
                    for c, cell in enumerate(row.cells):
                        print("Table[{}][{}] = {}".format(r, c, cell.text))

            f = open("{}/output_{}.txt".format(file_path, page_number), "a")
            f.write(text)
            page_number += 1
    except Exception as e:
        stacktrace = traceback.format_exc()
        print(stacktrace)

        raise e

def lambda_handler(event, context):
    try:

        logger.info(event)
        status_code = event["statusCode"]

        results = {
            "BucketName": None,
            "EventType": None,
            "JobId": None,
            "JobStatus": None,
            "ObjectKey": None
        }

        if status_code == 200:
            event_type = event["body"]["EventType"]

            if event_type and event_type != "ObjectRemoved:Delete":
                if event["body"]["JobId"] is not None:
                    logger.info("Get textract outputs")

                    bucket_name = event["body"]["BucketName"]
                    object_key = event["body"]["ObjectKey"]
                    job_id = event["body"]["JobId"]
                    job_status = event["body"]["JobStatus"]
                    file_name = object_key.split("/")[-1]

                    path = Path(os.path.join(output_file_path, job_id))
                    path.mkdir(parents=True, exist_ok=True)

                    extract_blocks(job_id, job_status, os.path.join(output_file_path, job_id))

                    chunks = get_chunks(file_name, object_key, os.path.join(output_file_path, job_id))

                    if len(object_key.split("/")) == 5:
                        index_name = object_key.split("/")[3]
                        new_es_url = es_url + "/" + es_index_name + "-" + index_name
                    else:
                        new_es_url = es_url + "/" + es_index_name

                    delete_index(new_es_url)

                    create_index(new_es_url)

                    index_documents(new_es_url, chunks)

                    results["BucketName"] = bucket_name
                    results["EventType"] = event_type
                    results["ObjectKey"] = object_key

                    return {
                        'statusCode': 200,
                        'body': results
                    }
                else:
                    raise Exception("No documents to index")
        else:
            return event
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e
