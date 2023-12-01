import openai
from data_models import HumanMessage, AIMessage, SystemMessage
import json
from utils import remove_responses
from marqo_search import search
from typing import List, Dict, Generator

from dotenv import load_dotenv

load_dotenv()

# GPT4 works much better than 3.5 for this however it is a lot more expensive

# GPT_MODEL = "gpt-3.5-turbo-0613"
GPT_MODEL = "gpt-4-0613"

FUNCTION_DESCRIPTION = "Search the Marqo index whenever Marqo is mentioned or you are told to search. This function returns information about how to use Marqo amongst other things."

SYSTEM_PROMPT = """
IMPORTANT: All code blocks should specify the language so that markdown can be rendered properly, e.g. ```python\n```.
Never make stuff up about the Marqo API, use the search functionality for it, stick to the facts.
"""

FUNCTIONS = [
    {
        "name": "search",
        "description": FUNCTION_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural language search query",
                },
            },
            "required": ["query"],
        },
    },
]


def format_chat(conversation: List[str], user_input: str) -> List[Dict[str, str]]:
    primer_message = [SystemMessage(content=SYSTEM_PROMPT)]
    llm_conversation = []
    approx_tokens = []
    for i in range(len(conversation)):
        if i % 2:
            msg = AIMessage(content=remove_responses(conversation[i]))
        else:
            msg = HumanMessage(content=conversation[i])
        approx_tokens.append(len(msg.content) // 4)
        llm_conversation.append(msg)

    # not particularly efficient or sophisticated, but it does the job
    # in reality what you probably want to do here is keep a running summary of the conversation
    # to cut down on tokens
    while sum(approx_tokens) > 3000:
        llm_conversation.pop(0)
        approx_tokens.pop(0)

    llm_conversation.append(HumanMessage(content=user_input))

    llm_conversation = primer_message + llm_conversation

    open_ai_conversation = [m.to_dict() for m in llm_conversation]
    return open_ai_conversation


def append_function_deltas(
    function_call: Dict[str, str], delta_function_call: Dict[str, str]
) -> Dict[str, str]:
    function_call["arguments"] += delta_function_call["arguments"]
    return function_call


def converse(user_input: str, conversation: List[str], limit: int) -> Generator:
    conversation = format_chat(conversation, user_input)
    stream1 = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=conversation,
        functions=FUNCTIONS,
        function_call="auto",
        stream=True,
    )
    function_call = None
    for chunk in stream1:
        token = chunk["choices"][0].get("delta", {}).get("content")
        if token is not None:
            yield token.encode("utf-8")
        elif (
            chunk["choices"][0]["delta"].get("function_call") and function_call is None
        ):
            yield "```curl\nBackend Function Call\n".encode("utf-8")
            func = chunk["choices"][0]["delta"]["function_call"]
            yield f"Function: {func['name']}\n".encode("utf-8")
            yield "Arguments:\n".encode("utf-8")
            yield func["arguments"].encode("utf-8")
            function_call = func
        elif chunk["choices"][0]["delta"].get("function_call"):
            func = chunk["choices"][0]["delta"]["function_call"]
            yield func["arguments"].encode("utf-8")
            function_call = append_function_deltas(function_call, func)

    if function_call is not None:
        yield "\n```\n".encode("utf-8")

    stream1.close()

    if function_call is None:
        return

    message = {
        "role": "assistant",
        "content": None,
        "function_call": {
            "name": function_call["name"],
            "arguments": function_call["arguments"],
        },
    }

    if message.get("function_call"):
        function_name = message["function_call"]["name"]

        arguments = json.loads(message["function_call"]["arguments"])
        function_response = search(query=arguments.get("query"), limit=limit)
        yield "\n```curl\nResponse:\n".encode("utf-8")
        yield f"{json.dumps(function_response, indent=4)}\n```\n".encode("utf-8")

        stream2 = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                *conversation,
                message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                },
            ],
            stream=True,
        )
        for chunk in stream2:
            token = chunk["choices"][0].get("delta", {}).get("content")
            if token is not None:
                yield token.encode("utf-8")
        stream2.close()
