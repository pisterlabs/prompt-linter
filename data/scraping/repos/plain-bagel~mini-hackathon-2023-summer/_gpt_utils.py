import os
from queue import Queue

import openai
from dotenv import load_dotenv
from openai.openai_object import OpenAIObject


# Load Environment variables and have them as constants
load_dotenv()

# Get key from .env file
openai.api_key = os.getenv("OPENAI_API_KEY")


def chat(messages: list[dict[str, str]], stream: bool = False, queue: Queue | None = None) -> OpenAIObject | None:
    """
    Generate text using ChatGPT as a separate process
    """
    try:
        gpt_res: OpenAIObject = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.8,
            max_tokens=1000,
            stream=stream,
            messages=messages,
        )
    except Exception as e:
        print(e)
        if queue is not None:
            queue.put(None)
        return None

    # Add to queue (for inter-process communication)
    if queue is not None:
        if not stream:
            queue.put(gpt_res["choices"][0]["message"])
        else:
            # Add chunks to queue when streaming
            for chunk in gpt_res:
                try:
                    queue.put(chunk["choices"][0]["delta"])
                except EOFError:
                    continue
    return gpt_res


def embed(prompt: str, queue: Queue | None = None) -> list[float] | None:
    """
    Generate embeddings using OpenAI Embedding API
    """
    prompt = prompt.replace("\n", " ")
    try:
        gpt_res: OpenAIObject = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=[prompt],
        )
        embedding: list[float] = gpt_res["data"][0]["embedding"]
    except Exception as e:
        print(e)
        if queue is not None:
            queue.put(None)
        return None

    # Add to queue (for inter-process communication)
    if queue is not None:
        queue.put(embedding)
    return embedding
