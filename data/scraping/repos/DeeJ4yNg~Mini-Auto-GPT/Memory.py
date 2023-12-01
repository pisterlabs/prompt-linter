from __future__ import annotations
import dataclasses
import os
from typing import Any,List
import numpy as np
import orjson
import openai
from openai.error import APIError, RateLimitError
import time

SAVE_OPTIONS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS
EMBED_DIM = 1536

def get_memory(init=False):
    memory = None
    if memory is None:
        memory = LocalCache(r"E:\AutoGPT\Test_AG\Test_AG\WindowsAssistant\Local_Memory.json")
        if init:
            memory.clear()
    return memory

def create_default_embeddings():
    return np.zeros((0, EMBED_DIM)).astype(np.float32)

@dataclasses.dataclass
class CacheContent:
    texts: List[str] = dataclasses.field(default_factory=list)
    embeddings: np.ndarray = dataclasses.field(
        default_factory=create_default_embeddings
    )

def create_embedding_with_ada(text) -> list:
    num_retries = 10
    for attempt in range(num_retries):
        backoff = 2 ** (attempt + 2)
        try:
            return openai.Embedding.create(
                    input=[text], model="text-embedding-ada-002"
                )["data"][0]["embedding"]
        except RateLimitError:
            pass
        except APIError as e:
            if e.http_status == 502:
                pass
            else:
                raise
            if attempt == num_retries - 1:
                raise
        time.sleep(backoff)

class LocalCache():

    def __init__(self, filename) -> None:
        self.filename = filename
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "w+b") as f:
                    file_content = f.read()
                    if not file_content.strip():
                        file_content = b"{}"
                        f.write(file_content)

                    loaded = orjson.loads(file_content)
                    self.data = CacheContent(**loaded)
            except orjson.JSONDecodeError:
                print(f"Error: The file '{self.filename}' is not in JSON format.")
                self.data = CacheContent()
        else:
            print(
                f"Warning: The file '{self.filename}' does not exist."
                "Local memory would not be saved to a file."
            )
            self.data = CacheContent()

    def add(self, text: str):
        if "Command Error:" in text:
            return ""
        self.data.texts.append(text)

        embedding = create_embedding_with_ada(text)

        vector = np.array(embedding).astype(np.float32)
        vector = vector[np.newaxis, :]
        self.data.embeddings = np.concatenate(
            [
                self.data.embeddings,
                vector,
            ],
            axis=0,
        )

        with open(self.filename, "wb") as f:
            out = orjson.dumps(self.data, option=SAVE_OPTIONS)
            f.write(out)
        return text

    def clear(self) -> str:
        self.data = CacheContent()
        return "Obliviated"

    def get(self, data: str) -> list[Any] | None:
        return self.get_relevant(data, 1)

    def get_relevant(self, text: str, k: int) -> list[Any]:
        embedding = create_embedding_with_ada(text)
        scores = np.dot(self.data.embeddings, embedding)
        top_k_indices = np.argsort(scores)[-k:][::-1]
        return [self.data.texts[i] for i in top_k_indices]

    def get_stats(self) -> tuple[int, tuple[int, ...]]:
        return len(self.data.texts), self.data.embeddings.shape
    
