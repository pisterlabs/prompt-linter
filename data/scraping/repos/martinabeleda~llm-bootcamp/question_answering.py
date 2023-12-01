"""Question ansering using embeddings-based search

This script implements a simple question answering pipeline based on pre-computed
embeddings. This is based on the tutorial from the OpenAI cookbook

See:
- https://cookbook.openai.com/examples/question_answering_using_embeddings
"""

import ast
from typing import Callable

import tiktoken
import pandas as pd
from loguru import logger
from openai import OpenAI
from scipy import spatial

from llm_bootcamp import utils

client = OpenAI(
    api_key="mabeleda",
    base_url="http://openai-api-proxy.discovery:8888/v1",
)

EMBEDDINGS_PATH = "https://cdn.openai.com/API/examples/data/winter_olympics_2022.csv"
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-4"


def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn: Callable = lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100,
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    logger.info(f"Creating embeddings for {query=} using {EMBEDDING_MODEL}")
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding

    strings_and_relatedness: list[tuple[str, float]] = [
        (row.text, relatedness_fn(query_embedding, row.embedding))
        for _, row in df.iterrows()
    ]
    strings_and_relatedness.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatedness[:top_n])

    return strings, relatednesses


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int,
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, _ = strings_ranked_by_relatedness(query, df)
    introduction = (
        "Use the below articles on the 2022 Winter Olympics to answer the subsequent question. "
        'If the answer cannot be found in the articles, write "I could not find an answer."'
    )
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    df: pd.DataFrame,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    logger.info(message)
    messages = [
        {
            "role": "system",
            "content": "You answer questions about the 2022 Winter Olympics.",
        },
        {"role": "user", "content": message},
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    response_message = completion.choices[0].message.content
    return response_message


def main():
    df = utils.read_csv_with_cache(EMBEDDINGS_PATH)
    df["embedding"] = df["embedding"].apply(ast.literal_eval)
    logger.info(df.head())

    query = "Which athletes won the gold medal in curling at the 2022 Winter Olympics?"

    response = ask(query, df)
    logger.info(response)


if __name__ == "__main__":
    main()
