import boto3
import json
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings


# Create bedrock boto3 clients
bedrock = boto3.client(service_name='bedrock')
bedrock_runtime = boto3.client(service_name='bedrock-runtime')
# Create bedrock_embeddings instance using LangChain
bedrock_embeddings = BedrockEmbeddings(client=bedrock_runtime)


def get_t2t_fms(vendors:list) -> list:
    """Generate a list of text-to-text FMs available with Bedrock, for specific vendors"""
    all_fms = bedrock.list_foundation_models()['modelSummaries']
    t2t_fms = []
    for fm in all_fms:
        if fm['inputModalities'] == ['TEXT'] and fm['outputModalities'] == ['TEXT']:
            if any(e in fm['modelId'] for e in vendors):
                t2t_fms.append(fm['modelId'])
    return t2t_fms


def ask_fm(modelid:str, prompt:str) -> str:
    """Invoke specific FM using boto3 and pass prompt and max tokens - all other inference parameters will use default values"""
    if "ai21.j2" in modelid:
        body = json.dumps({
            "prompt": prompt,
            "maxTokens": 2048})
    elif "anthropic.claude" in modelid:
        body = json.dumps({
            "prompt": f"\n\nHuman:{prompt}\n\nAssistant:",
            "max_tokens_to_sample": 2048})
    elif "cohere" in modelid:
        body = json.dumps({
            "prompt": prompt,
            "max_tokens": 2048})
    accept = "application/json"
    contentType = "application/json"
    # Invoke FM
    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelid, accept=accept, contentType=contentType
        )
    # Parse and print output
    response_body = json.loads(response["body"].read())
    if "ai21.j2" in modelid:
        return response_body["completions"][0]["data"]["text"]
    elif "anthropic.claude" in modelid:
        return response_body["completion"]
    elif "cohere" in modelid:
        return response_body["generations"][0]["text"]
    

def get_fm(modelid:str):
    """Return requested Bedrock FM (LangChain) with max tokens in output"""
    if "ai21.j2" in modelid:
        inference_parameters = {
            "maxTokens": 2048
        }
    elif "anthropic.claude" in modelid:
        inference_parameters = {
            "max_tokens_to_sample": 2048,
            "temperature": 0
        }
    elif "cohere" in modelid:
        inference_parameters = {
            "max_tokens": 2048
        }
    return Bedrock(model_id=modelid, client=bedrock_runtime, model_kwargs=inference_parameters)