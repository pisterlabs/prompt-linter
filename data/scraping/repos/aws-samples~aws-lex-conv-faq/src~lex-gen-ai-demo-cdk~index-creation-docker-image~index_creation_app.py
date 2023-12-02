import boto3
import json
from pathlib import Path

import logging
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import os
from llama_index import (
    LangchainEmbedding,
    GPTVectorStoreIndex,
    LLMPredictor,
    ServiceContext,
    Document,
    PromptHelper,
    download_loader
)

from langchain.embeddings import HuggingFaceEmbeddings

import logging
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ACCOUNT_ID = boto3.client('sts').get_caller_identity().get('Account')
INDEX_BUCKET = "lexgenaistack-created-index-bucket-"+ACCOUNT_ID
S3_BUCKET = "lexgenaistack-source-materials-bucket-"+ACCOUNT_ID
ENDPOINT_NAME = "huggingface-pytorch-sagemaker-endpoint"
DELIMITER = "\n\n\n"
LOCAL_INDEX_LOC = "/tmp/index_files"

def handler(event, context):
    event_record = event['Records'][0]
    if event_record['eventName'] == "ObjectCreated:Put":
        if ".txt" in event_record['s3']['object']['key'].lower() or ".pdf" in event_record['s3']['object']['key'].lower():
            source_material_key = event_record['s3']['object']['key']
            logger.info(f"Source file {source_material_key} found")
        else:
            logger.error("INVALID FILE, MUST END IN .TXT or .PDF")
            return
    else:
        logger.error("NON OBJECTCREATION INVOCATION")
        return
    
    s3_client = boto3.client('s3')
    try:
        s3_client.download_file(S3_BUCKET, source_material_key, "/tmp/"+source_material_key)
        logger.info(f"Downloaded {source_material_key}")
    except ClientError as e:
        logger.error(e)
        return "ERROR READING FILE"
    
    if ".pdf" in source_material_key.lower():
        PDFReader = download_loader("PDFReader", custom_path="/tmp/llama_cache")
        loader = PDFReader()
        documents = loader.load_data(file=Path("/tmp/"+source_material_key))
    else:
        with open("/tmp/"+source_material_key) as f:
            text_list = f.read().split(DELIMITER)
        logger.info(f"Reading text with delimiter {repr(DELIMITER)}")
        documents = [Document(t) for t in text_list]
    
    # define prompt helper
    max_input_size = 400  # set maximum input size
    num_output = 50  # set number of output tokens
    max_chunk_overlap = 0  # set maximum chunk overlap
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    # define our LLM
    llm_predictor = LLMPredictor(llm=CustomLLM())
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(cache_folder="/tmp/HF_CACHE"))
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper, embed_model=embed_model,
    )

    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist(persist_dir=LOCAL_INDEX_LOC)

    for file in os.listdir(LOCAL_INDEX_LOC):
        s3_client.upload_file(LOCAL_INDEX_LOC+"/"+file, INDEX_BUCKET, file) # ASSUMES IT CAN OVERWRITE, I.E. S3 OBJECT LOCK MUST BE OFF
 
    logger.info("Index successfully created")
    return

def call_sagemaker(prompt, endpoint_name=ENDPOINT_NAME):
    payload = {
        "inputs": prompt,
        "parameters": {
            "do_sample": False,
            # "top_p": 0.9,
            "temperature": 0.1,
            "max_new_tokens": 200,
            "repetition_penalty": 1.03,
            "stop": ["\nUser:", "<|endoftext|>", "</s>"]
        }
    }

    sagemaker_client = boto3.client("sagemaker-runtime")
    payload = json.dumps(payload)
    response = sagemaker_client.invoke_endpoint(
        EndpointName=endpoint_name, ContentType="application/json", Body=payload
    )
    response_string = response["Body"].read().decode()
    return response_string

def get_response_sagemaker_inference(prompt, endpoint_name=ENDPOINT_NAME):
    resp = call_sagemaker(prompt, endpoint_name)
    resp = json.loads(resp)[0]["generated_text"][len(prompt):]
    return resp

class CustomLLM(LLM):
    model_name = "tiiuae/falcon-7b-instruct"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = get_response_sagemaker_inference(prompt, ENDPOINT_NAME)
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"