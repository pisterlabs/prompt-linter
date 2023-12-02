import boto3
import json

from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings

bedrock_region = "us-west-2"
bedrock_endpoint_url = "https://prod.us-west-2.frontend.bedrock.aws.dev"


def get_bedrock_client():
    return boto3.client(
        service_name='bedrock',
        region_name=bedrock_region,
        endpoint_url=bedrock_endpoint_url,
        # aws_access_key_id=BEDROCK_ACCESS_KEY, # Task Role 을 이용 해서 접근
        # aws_secret_access_key=BEDROCK_SECRET_ACCESS_KEY # Task Role 을 이용 해서 접근
    )


def get_bedrock_model(model_id):
    bedrock_client = get_bedrock_client()
    return Bedrock(model_id=model_id, client=bedrock_client)


def get_bedrock_embeddings():
    bedrock_client = get_bedrock_client()
    return BedrockEmbeddings(client=bedrock_client)


def get_predict_from_bedrock_model(model_id: str, question: str):
    llm = get_bedrock_model(model_id=model_id)
    return llm.predict(question)


def get_predict_from_bedrock_client(model_id: str, prompt: str, parameters: dict):
    bedrock_client = get_bedrock_client()
    return bedrock_client.invoke_model(
        body=json.dumps({"inputText": prompt, "textGenerationConfig": parameters}),
        modelId=model_id, accept="application/json", contentType="application/json"
    )