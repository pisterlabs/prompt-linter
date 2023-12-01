import asyncio
import json
import math
import os
from itertools import islice
from typing import AsyncIterable, Iterable, Optional

import networkx as nx
import nltk
import numpy as np
import numpy.typing as npt
import openai
import pandas as pd
import tqdm
import typer
from asyncstdlib import map as amap
from asyncstdlib.functools import reduce as areduce
from graphviz import Digraph
from instructor.patch import wrap_chatcompletion
from pydantic import BaseModel, Field
from tenacity import retry, wait_random_exponential

from prompting_techniques import AsyncTyper, async_disk_cache, execute, format_prompt

np.random.seed(1)

nltk.download("punkt")

client = openai.AsyncOpenAI()
app = AsyncTyper()
sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")
book_1984 = open("./data/1984.txt", "r").read()

semaphore = asyncio.Semaphore(128)


class VectorDatabase(BaseModel):
    text: list[str]
    embeddings: npt.NDArray[np.float32]

    class Config:
        arbitrary_types_allowed = True

    def save_to_file(self, filename: str):
        # Convert NumPy array to a list for JSON serialization
        data = {"text": self.text, "embeddings": self.embeddings.tolist()}
        with open(filename, "w") as file:
            json.dump(data, file)

    @classmethod
    def load_from_file(cls, filename: str):
        with open(filename, "r") as file:
            data = json.load(file)
        # Convert list back to NumPy array
        data["embeddings"] = np.array(data["embeddings"], dtype=np.float32)
        return cls(**data)

    async def add_text(self, text: str) -> None:
        async with semaphore:
            embeddings_response = await client.embeddings.create(
                model="text-embedding-ada-002",
                input=text,
            )
            embedding: npt.NDArray[np.float32] = np.expand_dims(
                np.array(embeddings_response.data[0].embedding), axis=0
            )
            self.text.append(text)
            self.embeddings = np.concatenate([self.embeddings, embedding], axis=0)

    async def top_k(self, query: str, k: int = 10) -> list[str]:
        query_embedding_response = await client.embeddings.create(
            model="text-embedding-ada-002",
            input=query,
        )
        query_embedding: npt.NDArray[np.float32] = np.array(
            query_embedding_response.data[0].embedding
        )

        # cosine similarity, get top k
        similarity: npt.NDArray[np.float32] = np.dot(query_embedding, self.embeddings.T) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(self.embeddings, axis=1)
        )
        sorted_similarity_indices: npt.NDArray[np.int64] = np.argsort(similarity)[::-1]
        top_k: list[str] = [self.text[i] for i in sorted_similarity_indices[:k]]
        return top_k


@retry(wait=wait_random_exponential(multiplier=1, max=3))
async def get_ask_1984_response(vecdb: VectorDatabase, question: str) -> AsyncIterable[str | None]:
    related_passages = "\n".join(await vecdb.top_k(question, k=5))
    result = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": format_prompt(
                    """
                You are an AI question answer bot built with the knowledge and lessons from the famous book 1984 by George Orwell. \
                You have one goal: to answer questions and give advice about the book 1984. \
                
                Guideless:
                - You should answer the question directly and not provide any other information.
                - You should not provide any information that is not directly related to the question.
                - Keep your answers short and to the point.
                """
                ),
            },
            {
                "role": "system",
                "content": format_prompt(
                    f"""
                Here are some direct excerpts from the book 1984 related to the users question:
                
                {related_passages}
                """
                ),
            },
            {
                "role": "user",
                "content": f"""Here is the users question:
                
                {question}
                """,
            },
        ],
        model="gpt-4",
        temperature=0,
        seed=256,
        max_tokens=128,
        stream=True,
    )
    async for message in result:
        assert len(message.choices) > 0, "No choices were provided."
        content = message.choices[0].delta.content
        yield content


def sliding_window(iterable: list[str], window_size: int, stride: int) -> Iterable[list[str]]:
    """Generate a sliding window of specified size over the iterable."""
    total_iterations = math.ceil((len(iterable) - window_size) / stride) + 1
    with tqdm.tqdm(desc="Sliding Window", total=total_iterations) as progress:
        for i in range(0, len(iterable) - window_size + 1, stride):
            yield iterable[i : i + window_size]
            progress.update(1)

async def read_or_create_vecdb() -> VectorDatabase:
    ## CHeck if vecdb exists
    vecdb_filename = "./data/vecdb.json"
    if os.path.exists(vecdb_filename):
        vecdb = VectorDatabase.load_from_file(vecdb_filename)
    else:
        vecdb = VectorDatabase(text=[], embeddings=np.empty((0, 1536), dtype=np.float32))

        window_size = 16
        stride = 8
        sentences: list[str] = sent_detector.tokenize(book_1984)  # type: ignore

        def join_sentences(sentences: list[str]) -> str:
            return " ".join(sentences)

        chunks = [i for i in map(join_sentences, sliding_window(sentences, window_size, stride))]
        await execute([vecdb.add_text(chunk) for chunk in chunks], desc="Adding to vecdb")
        vecdb.save_to_file(vecdb_filename)
    return vecdb


@app.command()
async def ask_1984():
    vecdb = await read_or_create_vecdb()
    text: str = str(typer.prompt("What question / advice do you want to ask about the book 1984?", type=str))
    assert len(text) > 0, "Please provide some text."
    typer.echo("\n")
    async for token in get_ask_1984_response(vecdb, text):
        typer.echo(token, nl=False)

@app.command()
async def vec_lookup(n: int = 3):
    vecdb = await read_or_create_vecdb()
    text: str = str(typer.prompt(f"Query top {n} results from the vecdb", type=str))
    assert len(text) > 0, "Please provide some text."
    typer.echo("\n")
    for passage in await vecdb.top_k(text, k=n):
        typer.echo(passage)
        typer.echo("\n")
        

if __name__ == "__main__":
    app()
