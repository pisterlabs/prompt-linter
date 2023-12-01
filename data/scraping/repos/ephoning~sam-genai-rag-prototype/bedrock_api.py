import json
import os
import sys

import boto3
import botocore

from langchain.embeddings import BedrockEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

import bedrock


from constants import *

# ---- ⚠️ Un-comment and edit the below lines as needed for your AWS setup ⚠️ ----
# os.environ["AWS_DEFAULT_REGION"] = "<REGION_NAME>"  # E.g. "us-east-1"
# os.environ["AWS_PROFILE"] = "<YOUR_PROFILE>"
# os.environ["BEDROCK_ASSUME_ROLE"] = "<YOUR_ROLE_ARN>"  # E.g. "arn:aws:..."


default_embeddings_model_id = os.environ["DEFAULT_EMBEDDINGS_MODEL_ID"]
default_model_id = os.environ["DEFAULT_MODEL_ID"]


def get_bedrock_client():
    boto3_bedrock = bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        region=os.environ.get("AWS_DEFAULT_REGION", None))
    return boto3_bedrock


def get_langchain_client(bedrock_client, model_id=None, inference_modifier=None):
    if not model_id:
        model_id = default_model_id
    if not inference_modifier:
        inference_modifier = DEFAULT_INFERENCE_MODIFIER
        
    lc_client = Bedrock(
        model_id=model_id,
        client=bedrock_client,
        model_kwargs=inference_modifier
    )
    return lc_client
        

def get_embeddings_client(bedrock_client, model_id=None):
    if not model_id:
        model_id = default_embeddings_model_id
        
    embeddings_client = BedrockEmbeddings(model_id=model_id, client=bedrock_client)
    return embeddings_client


def get_conversation_memory():
    conversation_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return conversation_memory


def get_num_tokens(lc_client, text):
    return lc_client.get_num_tokens(text)
    

def compose_prompt(model_id, params):
    prompt_template = PROMPT_TEMPLATES.get((model_id, params['use_case']))
    if not prompt_template:
        prompt_template = PROMPT_TEMPLATES[(model_id, None)]
    if not params.get('txt'):
        params['txt'] = PROMPT_TEXTS[params['use_case']]
    prompt = prompt_template.format(**params)
    # allow for multiple nested parameterized string to be expanded / flattened
    for i in range(0,3): 
        prompt = prompt.format(**params)
    return prompt

        
def compose_body(model_id, prompt=None, params=None):
    if prompt:
        params = {'use_case': None, 
                  'txt': prompt}
    prompt = compose_prompt(model_id, params)
    body_composer = BODY_COMPOSERS.get(model_id)
    body = body_composer(prompt)
    return body

        
def invoke_model(bedrock_client, model_id=None, prompt=None, params=None):
    if not model_id:
        model_id = default_model_id

    body = compose_body(model_id, prompt, params)
    accept = 'application/json'
    contentType = 'application/json'
    response_parser = RESPONSE_PARSERS[model_id]
    try:
        response = bedrock_client.invoke_model(body=body, modelId=model_id, accept=accept, contentType=contentType)
        response_body = json.loads(response.get('body').read())
        output = response_parser(response_body)
        return output

    except botocore.exceptions.ClientError as error:

        if error.response['Error']['Code'] == 'AccessDeniedException':
            return f"\x1b[41m{error.response['Error']['Message']}\
                    \nTo troubeshoot this issue please refer to the following resources.\
                     \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                     \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n"
        else:
            raise error

            
def invoke_model_langchain(lc_client, prompt=None, params=None):
    if not prompt:
        prompt = compose_prompt(lc_client.model_id, params)    
        
    response = lc_client(prompt)
    return response

            
def invoke_model_stream(bedrock_client, model_id=None, prompt=None, params=None):
    if not model_id:
        model_id = default_model_id

    body = compose_body(model_id, prompt, params)
    accept = 'application/json'
    contentType = 'application/json'
    response_parser = RESPONSE_PARSERS[model_id]

    try:
        response = bedrock_client.invoke_model_with_response_stream(body=body, modelId=model_id, accept=accept, contentType=contentType)
        stream = response.get('body')
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    chunk_obj = json.loads(chunk.get('bytes').decode())
                    text = response_parser(chunk_obj)
                    yield text
        else:
            yield '<did not receive any stream output>'

    except botocore.exceptions.ClientError as error:

        if error.response['Error']['Code'] == 'AccessDeniedException':
            return f"\x1b[41m{error.response['Error']['Message']}\
                    \nTo troubeshoot this issue please refer to the following resources.\
                     \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                     \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n"
        else:
            raise error


def invoke_chunked_summarization_langchain(lc_client, text, chunk_size=4000, chunk_overlap=100, chain_type='map_reduce'):
    """
    chain_type options: 'chunk' | 'map_reduce' | 'refine'
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.create_documents([text])
    summary_chain = load_summarize_chain(llm=lc_client, chain_type=chain_type, verbose=False)
    try:
        output = summary_chain.run(docs)
        return output

    except ValueError as error:
        if  "AccessDeniedException" in str(error):
            return f"\x1b[41m{error}\
            \nTo troubeshoot this issue please refer to the following resources.\
             \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
             \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n"  
        else:
            raise error    
            
            
