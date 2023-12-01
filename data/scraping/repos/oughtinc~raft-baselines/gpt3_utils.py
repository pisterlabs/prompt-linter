import openai
from dotenv import load_dotenv
import os
import time
from cachetools import cached, LRUCache
from typing import List, Dict, Tuple, Any, cast

from raft_baselines.utils.tokenizers import TransformersTokenizer

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@cached(cache=LRUCache(maxsize=1e9))
def complete(
    prompt: str,
    engine: str = "ada",
    max_tokens: int = 5,
    temperature: float = 1.0,
    top_p: float = 1.0,
    n: int = 1,
    echo: bool = False,
    stop: Tuple[str, ...] = ("\n",),
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
):
    openai_completion_args = dict(
        api_key=OPENAI_API_KEY,
        engine=engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
        logprobs=100,  # Always request 100 so can easily count tokens in completion
        echo=echo,
        stop=stop,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )

    success = False
    retries = 0
    while not success:
        try:
            response = openai.Completion.create(**openai_completion_args)
            success = True
        except Exception as e:
            print(f"Exception in OpenAI completion: {e}")
            retries += 1
            if retries > 3:
                raise Exception("Max retries reached")
                break
            else:
                print("retrying")
                time.sleep(retries * 15)

    return cast(Dict[str, Any], response)


@cached(cache=LRUCache(maxsize=1e9))
def search(
    documents: Tuple[str, ...], query: str, engine: str = "ada"
) -> List[Dict[str, Any]]:
    response = None
    error = None
    tokenizer = TransformersTokenizer("gpt2")
    query = tokenizer.truncate_by_tokens(query, 1000)
    short_enough_documents = [
        tokenizer.truncate_by_tokens(document, 2034 - tokenizer.num_tokens(query))
        for document in documents
    ]

    success = False
    retries = 0
    while not success:
        try:
            response = openai.Engine(engine, api_key=OPENAI_API_KEY).search(
                documents=short_enough_documents, query=query
            )
            success = True
        except Exception as e:
            print(f"Exception in OpenAI search: {e}")
            retries += 1
            if retries > 3:
                raise Exception("Max retries reached")
                break
            else:
                print("retrying")
                time.sleep(retries * 15)

    assert response is not None
    results = response["data"]

    return results
