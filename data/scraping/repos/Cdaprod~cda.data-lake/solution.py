#!/usr/bin/env python3


from langchain.tools import Tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableBranch

from typing import Any, Dict

import boto3

import os
import logging

from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer

from weaviate import Client
from minio import Minio
from minio.error import MinioException
from clients import initialize_weaviate_client, initialize_minio_client

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level="INFO")
logger.propagate = False  # Prevent logger from propagating to the root logger

# Define the Pydantic models
class ImprovementIssue(BaseModel):
    improvement_summary: str
    improvement_details: str
    improvement_code: str

class DocumentationIssue(BaseModel):
    documentation_summary: str
    documentation_details: str
    documentation_code: str

# Define the GithubTool class
class GithubTool:
    def __init__(self):
        pass

    async def _classify_issue(self, issue_message: str) -> str:
        logger.info(f"Classifying issue: {issue_message}")
        classify_issue_chain = ChatPromptTemplate.from_template(
            """
            Given the issue message below, classify it as either `Improvement`, `Documentation`, or `Other`.
            <issue_message>
            {issue_message}
            </issue_message>
            Classification:
            """
        ) | ChatOpenAI(model="gpt-4", temperature=0)
        
        issue_type = await classify_issue_chain.run({"issue_message": issue_message})
        return issue_type
    
    async def _generate_solution(self, issue_type: str, issue_message: str) -> Dict[str, Any]:
        if issue_type == "Improvement":
            return ImprovementIssue(
                improvement_summary="Summary here",
                improvement_details="Details here",
                improvement_code="Code here"
            ).dict()
        elif issue_type == "Documentation":
            return DocumentationIssue(
                documentation_summary="Summary here",
                documentation_details="Details here",
                documentation_code="Code here"
            ).dict()

    async def solve_issue(self, issue_message: str) -> Dict[str, Any]:
        logger.info(f"Solving issue: {issue_message}")
        issue_type = await self._classify_issue(issue_message)
        solution = await self._generate_solution(issue_type, issue_message)
        logger.info(f"Issue solved with solution: {solution}")
        return solution

# Environment Checks
def check_env():
    logger.info("Checking environment variables.")
    required_vars = ['GH_TOKEN', 'REPO', 'BRANCH', 'MAIN_BRANCH']
    missing_vars = [var for var in required_vars if var not in os.environ]
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        raise EnvironmentError(f"Missing environment variables: {', '.join(missing_vars)}")
    logger.info("Environment variables checked.")

def generate_text_embeddings(text):
    logger.info("Generating text embeddings.")
    
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        inputs = tokenizer(text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    except Exception as e:
        logger.error(f"Tokenizer failed: {e}")
        return None
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=8)
        outputs = model(**inputs)
    except Exception as e:
        logger.error(f"Model failed: {e}")
        return None
    
    try:
        pooled_output = outputs.pooler_output
        embeddings = pooled_output.detach().cpu().numpy()
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None
    
    logger.info("Text embeddings generated successfully.")
    return embeddings

# Create custom object in Weaviate
def create_custom_object(embeddings):
    logger.info("Creating custom object in Weaviate.")
    client = initialize_weaviate_client()
    className = "MyCustomClass"
    client.create_class(className)
    properties = {
        "text": {"type": "string"},
        "embeddings": {"type": "vector", "dimensions": 768},
        "code": {"type": "string"}
    }
    client.update_class(className, properties)
    objName = "MyCustomObject"
    objProperties = {
        "text": "This is my custom object.",
        "embeddings": embeddings,
        "code": ""
    }
    client.create_object(objName, className, objProperties)
    logger.info(f"Custom object created: {objName}")
    return objName

def process_and_index_object(key, s3, tokenizer, model, client):
    logger.info(f"Processing and indexing object with key: {key}")
    
    file_path = f"s3://{os.environ['MINIO_ENDPOINT']}/{key}"
    try:
        with open(file_path, 'r') as f:
            text = f.read()
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        return
    
    try:
        embeddings = generate_text_embeddings(text)
    except Exception as e:
        logger.error(f"Failed to generate text embeddings: {e}")
        return
    
    try:
        obj_name = create_custom_object(embeddings)
    except Exception as e:
        logger.error(f"Failed to create custom object: {e}")
        return
    
				logger.info(f"Successfully created custom object: {obj_name}")

def object_tool(inp: str) -> str:
	 	logger.info(f"Processing object: {inp}")
    check_env()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=8)
    client = Client("http://192.168.0.25:8082")
    s3 = initialize_minio_client()
    process_and_index_object(inp, s3, tokenizer, model, client)
    logger.info(f"Object processed and indexed: {inp}")
    return f"Processed and indexed object: {inp}"

def define_tools():
    github_tool = GithubTool()
    search_tool = Tool(
        name="Search",
        func=github_tool.solve_issue,  # Use the solve_issue method from GithubTool
        description="A tool to solve GitHub issues"
    )
    object_tools = [
        Tool(
            name="ObjectTool",
            func=object_tool,
            description="A function to process and index objects based on unstructured data"
        )
    ]
    ALL_TOOLS = [search_tool] + object_tools
    return ALL_TOOLS