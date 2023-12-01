import boto3
from langchain.agents import AgentType, initialize_agent, tool
from langchain.llms import Bedrock
from langchain.tools import StructuredTool


from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from typing import List, Dict, Any
import boto3

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

llm = Bedrock(client=bedrock_runtime, model_id="anthropic.claude-v2")

# Demo tools
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)


def adder(a: float, b: float) -> float:
    """Multiply the provided floats."""
    return a + b


def multiplier(a: float, b: float) -> float:
    """Multiply the provided floats."""
    # TODO: Implement this function
    return None


def get_current_time() -> str:
    """Returns the current time."""
    # TODO: Implement this function
    return None

# AWS Well-Architected Framework tool
def well_arch_tool(query: str) -> Dict[str, Any]:
    """Returns text from AWS Well-Architected Framework releated to the query"""
    embeddings = BedrockEmbeddings(
        client=bedrock_runtime,
        model_id="amazon.titan-embed-text-v1",
    )
    vectorstore = FAISS.load_local("local_index", embeddings)
    docs = vectorstore.similarity_search(query)

    resp_json = {"docs": docs}

    return resp_json


adder_tool = StructuredTool.from_function(adder)
time_tool = StructuredTool.from_function(get_current_time)
multiplier_tool = StructuredTool.from_function(multiplier)
word_tool = StructuredTool.from_function(get_word_length)
aws_well_arch_tool = StructuredTool.from_function(well_arch_tool)

# Start Agent
agent_executor = initialize_agent(
    [multiplier_tool, word_tool, aws_well_arch_tool, adder_tool, time_tool],
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Run some queries
resp = agent_executor.run("How many letters is in 'educa'?")
print(resp)
resp = agent_executor.run("What is 3 plus 4?")
print(resp)
resp = agent_executor.run(
    "What does the AWS Well-Architected Framework say about how to create secure VPCs?"
)
print(resp)
resp = agent_executor.run("What is the current time?")
print(resp)
resp = agent_executor.run("What is 3 times 4?")
print(resp)