"""
Combine documents and question in a prompt and send it to an LLM to get the answer. 
"""
import openai
from promptflow import tool
from promptflow.connections import AzureOpenAIConnection
from typing import Generator
import os

@tool
def rag(
    system_prompt: str,
    chat_history: list[str],
    query: str,
    azure_open_ai_connection: AzureOpenAIConnection,
    deployment_name: str
) -> Generator[str, None, None]:
    """
    Ask the LLM to answer the user's question given the chat history and context.
    """
    openai.api_type = azure_open_ai_connection.api_type
    openai.api_base = azure_open_ai_connection.api_base
    openai.api_version = azure_open_ai_connection.api_version
    openai.api_key = azure_open_ai_connection.api_key

    messages = [{"role": "system", "content": system_prompt}]
    for item in chat_history:
        messages.append({"role": "user", "content": item["inputs"]["question"]})
        messages.append({"role": "assistant", "content": item["outputs"]["answer"]})
    messages.append({"role": "user", "content": query})

    chat_completion = openai.ChatCompletion.create(
        deployment_id=deployment_name,
        messages=messages,
        temperature=0,
        max_tokens=1024,
        n=1,
        stream=True
    )

    for chunk in chat_completion:
        if chunk["object"] == "chat.completion.chunk":
            if "content" in chunk["choices"][0]["delta"]:
                yield chunk["choices"][0]["delta"]["content"]

