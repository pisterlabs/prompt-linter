import openai
from data_models import HumanMessage, AIMessage, SystemMessage
import json
from utils import remove_responses
from knowledge_store import MarqoKnowledgeStore
from typing import List, Dict, Generator

from dotenv import load_dotenv

load_dotenv()

FUNCTIONS = [
    {
        "name": "search_marqo",
        "description": "This is a search engine, use it when the user instructs you to search in any way. Also use it for organisational knowledge or personal information.",
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


def search_marqo(query: str, mks: MarqoKnowledgeStore, limit: int) -> str:
    try:
        results = mks.query_for_content(query, limit=limit if limit is not None else 4)
        return json.dumps(results)
    except Exception as e:
        return {"marqo_search_error": e}


def format_chat(conversation: List[str], user_input: str) -> List[Dict[str, str]]:
    llm_conversation = [
        SystemMessage(
            content="All code should specify the language so that markdown can be rendered."
        )
    ]
    for i in range(len(conversation)):
        if i % 2:
            msg = AIMessage(content=remove_responses(conversation[i]))
        else:
            msg = HumanMessage(content=conversation[i])
        llm_conversation.append(msg)

    llm_conversation.append(HumanMessage(content=user_input))
    open_ai_conversation = [vars(m) for m in llm_conversation]
    return open_ai_conversation


def append_function_deltas(
    function_call: Dict[str, str], delta_function_call: Dict[str, str]
) -> Dict[str, str]:
    function_call["arguments"] += delta_function_call["arguments"]
    return function_call


def converse(
    user_input: str, conversation: List[str], mks: MarqoKnowledgeStore, limit: int
) -> Generator:
    conversation = format_chat(conversation, user_input)
    stream1 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
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
        function_response = search_marqo(
            query=arguments.get("query"), mks=mks, limit=limit
        )
        yield "\n```curl\nResponse:\n".encode("utf-8")
        yield f"{json.dumps(function_response, indent=4)}\n```\n".encode("utf-8")

        stream2 = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
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
