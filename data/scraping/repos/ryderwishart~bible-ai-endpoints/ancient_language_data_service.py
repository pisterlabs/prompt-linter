from pathlib import Path
from modal import Image, Secret, Stub, web_endpoint

image = Image.debian_slim().pip_install(
    "openai~=0.27.4",
    "tiktoken==0.3.0",
    "langchain~=0.0.216",
)
stub = Stub(
    name="ancient-language-data-service",
    image=image,
    secrets=[Secret.from_name("openai-qa")],
)

import requests
import logging
from langchain.agents import initialize_agent, Tool, tool
from langchain.agents import AgentType
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Any, Dict, List, Generator
from fastapi.responses import StreamingResponse
from langchain.callbacks.base import AsyncCallbackHandler
import time

class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self):
        self.generated_tokens = []

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.generated_tokens.append(token)

    def stream_tokens(self) -> Generator[str, None, None]:
        while True:
            while self.generated_tokens:
                yield self.generated_tokens.pop(0)
            time.sleep(0.05) # adjust this sleep time as needed

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

logger.info("Initializing agents...")

# Plan and execute agent to handle requests

endpoints = {
    "atlas": {
        "endpoint": "https://ryderwishart--atlas-agent-answer-question-using-atlas.modal.run/",
        "example": "https://ryderwishart--atlas-agent-answer-question-using-atlas.modal.run/?query=What%20are%20the%20discourse%20features%20of%20JHN%201:11?",
    },
    "syntax-bracket": {
        "endpoint": "https://ryderwishart--syntax-agent-get-syntax-for-query.modal.run/",
        "example": "https://ryderwishart--syntax-agent-get-syntax-for-query.modal.run/?query=Jesus%20in%201%20John%201:1",
    },
    "syntax-agent": {
        "endpoint": "https://ryderwishart--syntax-agent-syntax-qa-chain.modal.run/",
        "example": "https://ryderwishart--syntax-agent-syntax-qa-chain.modal.run/?query=What%20is%20the%20subject%20of%20the%20verb%20%27%20%27love%27%27%20in%20Matthew%205%3A44%3F",
    },
    "tyndale-agent": {
        "endpoint": "https://ryderwishart--tyndale-chroma-question.modal.run/",
        "example": "https://ryderwishart--tyndale-chroma-question.modal.run/?query=Who%20is%20Bartholomew?",
    },
    "tyndale-documents": {
        "endpoint": "https://ryderwishart--tyndale-chroma-get-documents.modal.run/",
        "example": "https://ryderwishart--tyndale-chroma-get-documents.modal.run/?query=Who%20is%20Bartholomew?",
    },
}


# Let's make one function tool per endpoint
@tool
def query_atlas(query: str):
    """Ask a question of the atlas endpoint."""
    url_encoded_query = query.replace(" ", "%20")
    endpoint, example = endpoints["atlas"].values()
    url = f"{endpoint}?query={url_encoded_query}"
    
    try:
        response = requests.get(url)
        return response.json()
    except:
        return {"error": "There was an error with the request. Please reformat request or try another tool."}


@tool
def query_syntax_bracket(query: str):
    """Ask a question of the syntax-bracket endpoint."""
    url_encoded_query = query.replace(" ", "%20")
    endpoint, example = endpoints["syntax-bracket"].values()
    url = f"{endpoint}?query={url_encoded_query}"
    response = requests.get(url)
    return response.json()


@tool
def query_syntax_agent(query: str):
    """Ask a question of the syntax-agent endpoint."""
    url_encoded_query = query.replace(" ", "%20")
    endpoint, example = endpoints["syntax-agent"].values()
    url = f"{endpoint}?query={url_encoded_query}"
    response = requests.get(url)
    return response.json()


@tool
def query_tyndale_agent(query: str):
    """Ask a question of the tyndale-agent endpoint."""
    url_encoded_query = query.replace(" ", "%20")
    endpoint, example = endpoints["tyndale-agent"].values()
    url = f"{endpoint}?query={url_encoded_query}"
    response = requests.get(url)
    return response.json()


@tool
def query_tyndale_documents(query: str):
    """Ask a question of the tyndale-documents endpoint."""
    url_encoded_query = query.replace(" ", "%20")
    endpoint, example = endpoints["tyndale-agent"].values()
    url = f"{endpoint}?query={url_encoded_query}"    
    response = requests.get(url)
    return response.json()


models = ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo"]
llm = ChatOpenAI(model=models[1], streaming=True, callbacks=[StreamingStdOutCallbackHandler(), StreamingLLMCallbackHandler()], temperature=0)


tools = [
    Tool(
        name = "Query Atlas",
        func=query_atlas.run,
        description="useful for when you need to answer questions about New Testament discourse analysis. You may include a Bible verse such as 'Matthew 5:44' or '1 John 3:1' in your query, or ask about a discourse feature."
    ),
    Tool(
        name = "Query Syntax Bracket",
        func=query_syntax_bracket.run,
        description="useful for when you need to view a syntax analysis of a given Bible verse. You must include a Bible verse in your query."
    ),
    Tool(
        name = "Query Syntax Agent",
        func=query_syntax_agent.run,
        description="useful for when you need to answer questions about New Testament syntax. You must include a Bible verse such as 'Matthew 5:44' or '1 John 3:1' in your query."
    ),
    Tool(
        name = "Query Tyndale Agent",
        func=query_tyndale_agent.run,
        description="useful for when you need to answer encyclopedic questions about Bible",
    ),
    Tool(
        name = "Query Tyndale Documents",
        func=query_tyndale_documents.run,
        description="useful for when you need to access encyclopedic documents related to a given query directly. Try this is the Query Tyndale Agent is not finding the right results.",
    ),
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)
planning_agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)


# @stub.function(keep_warm=1)
# @web_endpoint()
# def question(query: str):
#     """Ask a question of the agent."""
#     logger.info(f"Query: {query}")
#     result = planning_agent.run(query)
#     logger.info(f"Result: {result}")
#     return result





callback_handler = StreamingLLMCallbackHandler()

@stub.function(keep_warm=1)
@web_endpoint()
def question(query: str) -> StreamingResponse:
    """Ask a question of the agent."""
    return StreamingResponse(
        agent.arun(query, callback_handler=callback_handler), 
        media_type="text/html"
    )
