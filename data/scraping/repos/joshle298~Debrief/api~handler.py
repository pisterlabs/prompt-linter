import json
import os
import subprocess
import boto3
import nomic
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import AtlasDB
from langchain.embeddings.openai import OpenAIEmbeddings
import dotenv
dotenv.load_dotenv()

s3_client = boto3.client('s3')

ATLAS_TOKEN = os.getenv("ATLAS_TEST_API_KEY")

def handler(event, context):

    
    embeddings = OpenAIEmbeddings()
    db = AtlasDB("headline_data", embeddings, ATLAS_TOKEN, is_public=True)

    docs = db.similarity_search("trump")

    print(json.dumps(docs, indent=2))
    query = "haiku about loving me"
    body = {
        "message": "hello",
        "query": query,
        "input": event,
        "poem": run_agent("haiku about loving me"),
    }
    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }
    return response

def run_agent(query):
    """Builds and executes langchain agent"""
    llm = ChatOpenAI()
    return llm.call_as_llm(query)

