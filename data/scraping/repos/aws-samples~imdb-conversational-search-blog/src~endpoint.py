from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings, SentenceTransformerEmbeddings
from langchain import SagemakerEndpoint

import boto3

from src.content_handlers import AI21SageMakerContentHandler


def launch_encoder(model_name="gtr-t5-large"):
    """
    Launch sentence transformer embedding model
    Args:
        model_name(str): specific sentence transformer model
    Returns:
        SentenceTransformerEmbeddings: embedding model
    """
    return SentenceTransformerEmbeddings(model_name=model_name)


def amazon_bedrock_embeddings():
    """
    Launch bedrock embedding model
    Return:
        langchain.embeddings.BedrockEmbeddings: bedrock embedding model
    """
    return BedrockEmbeddings()


def create_bedrock_body(temp=0.0, topP=1, stop_sequences=["Human:"]):
    """
    Configurations for the bedrock llm model
    Args:
        temp(float): variability of model results
        topP(integer): control how deterministic the model is
        stop_sequences(list): stop generating text at these specific words
    """
    body = {
        "max_tokens_to_sample": 300,
        "temperature": temp,
        "top_k": 250,
        "top_p": topP,
        "stop_sequences": stop_sequences,
    }
    return body


def amazon_bedrock_llm(
    region="us-east-1", modelId="anthropic.claude-v1", verbose=False
):
    """
    Create bedrock llm from langchain. Make sure to have bedrock capabilities in this account
    Args:
        region(str): AWS region
        modelId(str): model type
    Returns:
        langchain.llms.Bedrock: bedrock llm
    """

    llm = Bedrock(region_name=region, model_id=modelId)
    llm.client = boto3.client(
        service_name="bedrock",
        region_name=region,
        endpoint_url=f"https://bedrock.{region}.amazonaws.com",
    )
    llm.model_kwargs = create_bedrock_body()
    return llm


def sagemaker_endpoint_ai21(endpoint_name):
    """
    Create langchain llm from a J2 Jumbo Instruct Endpoint within SM
    Args:
        endpoint_name(str): name of the J2 Jumbo Instruct endpoint
    Returns:
        Langchain llm
    """
    return SagemakerEndpoint(
        endpoint_name=endpoint_name,
        region_name="us-east-1",
        model_kwargs={"temperature": 0, "maxTokens": 300, "numResult": 1},
        content_handler=AI21SageMakerContentHandler(),
    )
