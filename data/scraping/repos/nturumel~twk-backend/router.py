import logging
from typing import List

import cachetools
from flask import Flask, jsonify, request
from langchain.tools import Tool

from twk_backend.custom_chat_agent.custom_chat_agent import CustomChatAgent
from twk_backend.custom_chat_agent.example_refine_chain import ExampleRefineChain
from twk_backend.tools.utils import get_tool

# Set up root logger to write messages with level INFO or higher to stdout.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app.log",
)

app = Flask(__name__)

# Create a TTLCache object with a maximum size and a time to live (in seconds)
chat_agent_cache: cachetools.TTLCache[str, CustomChatAgent] = cachetools.TTLCache(
    ttl=5 * 60, maxsize=5000
)

# Create a TTLCache object with a maximum size and a time to live (in seconds)
agent_samples_cache: cachetools.TTLCache[str, ExampleRefineChain] = cachetools.TTLCache(
    ttl=5 * 60, maxsize=5000
)


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/initialiseAgent", methods=["POST"])
def handle_initialize_agent():
    """
    Returns success or failure
    Initialise the chat agent and persist in cache

    :param sessionId: id of session
    :type str
    :param toolList
    :type list of tools
    :param agent
    :type Agent

    :return: success or failure for request
    :rtype: Response
    """
    data = request.get_json()
    session_id = data["sessionId"]

    # Check if session already initialised
    if session_id in chat_agent_cache:
        logging.info(f"Agent already initialised for input: {data}")
        return jsonify({"sessionId": session_id}), 200

    try:
        logging.info(f"Received input: {data}")

        # Extract components
        agent = data["agent"]
        samples = data.get("samples")
        tool_list = data["toolList"]
        logging.info("Components extracted")

        # Setup tools
        tools: List[Tool] = []
        for tool in tool_list:
            tool_type = tool["toolType"]
            tool_config = tool["toolConfig"]
            tool = get_tool(tool_type=tool_type, tool_config=tool_config)
            tools.append(tool)
        logging.info("Tool List Initialised")

        chat_agent = CustomChatAgent(
            name=agent["name"],
            writingStyle=agent.get("writingStyle") or "",
            temperature=agent["temperature"],
            prompt=agent["prompt"],
            tools=tools,
            samples=samples,
        )
        chat_agent_cache[session_id] = chat_agent
        return jsonify({"sessionId": session_id}), 200
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"sessionId": session_id, "error": str(e)}), 500


@app.route("/chatWithAgent", methods=["POST"])
def handle_chat_with_agent():
    """
    Returns agent response, success or failure for request

    :param sessionId
    :type str
    :param query
    :type str
    :return: Returns answer from chat agent
    :rtype: Response
    """
    data = request.get_json()  # get the request data (as JSON)
    session_id = data["sessionId"]
    try:
        # Extract
        query = data["query"]
        logging.info(f"Query: {query}")

        # Get agent
        chat_agent = chat_agent_cache[session_id]
        logging.info("Obtained chat agent")

        response = chat_agent.chat(query)
        logging.info(f"response: {response}")

        return jsonify(response), 200
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"sessionId": session_id, "error": str(e)}), 500


@app.route("/initaliseSamples", methods=["POST"])
def handle_intialise_samples():
    """
    Returns success or failure
    Initialise the chat agent samples and persist in cache

    :param sessionId: id of session
    :type str
    :param samples
    :type list of samples

    :return: success or failure for request
    :rtype: Response
    """
    data = request.get_json()  # get the request data (as JSON)
    session_id = data["sessionId"]

    # Check if session already initialised
    if session_id in agent_samples_cache:
        logging.info(f"Samples already initialised for input: {data}")
        return jsonify({"sessionId": session_id}), 200

    try:
        logging.info(f"Received input: {data}")

        # Extract components
        samples = data.get("samples")
        logging.info("Samples extracted")

        sample_refine_chain = ExampleRefineChain(
            examples=samples,
        )
        agent_samples_cache[session_id] = sample_refine_chain
        return jsonify({"sessionId": session_id}), 200
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"sessionId": session_id, "error": str(e)}), 500


@app.route("/refineAnswer", methods=["POST"])
def handle_refine_response_with_samples():
    """
    Returns agent response, success or failure for request

    :param sessionId
    :type str
    :param query
    :type str
    :return: Returns answer from chat agent
    :rtype: Response
    """
    data = request.get_json()  # get the request data (as JSON)
    session_id = data["sessionId"]
    try:
        # Extract
        query = data["query"]
        logging.info(f"Query: {query}")
        response = data["response"]
        logging.info(f"Response: {response}")

        # Get agent
        sample_refine_chain = None
        if session_id in agent_samples_cache:
            sample_refine_chain = agent_samples_cache[session_id]
            logging.info("Obtained chat agent samples")
        else:
            # Extract components
            samples = data.get("samples")
            logging.info("Samples extracted")

            sample_refine_chain = ExampleRefineChain(
                examples=samples,
            )
            agent_samples_cache[session_id] = sample_refine_chain

        refined_response = sample_refine_chain.refine_response(input=query, response=response)
        logging.info(f"response: {refined_response}")

        return jsonify(refined_response), 200
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"sessionId": session_id, "error": str(e)}), 500

