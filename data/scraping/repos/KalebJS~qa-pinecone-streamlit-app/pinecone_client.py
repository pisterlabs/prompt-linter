import json
from typing import Union

import openai
import pinecone
import tiktoken
import torch
from sentence_transformers import SentenceTransformer

from config import CONTEXT_PROMPT_NOTES, ENCODING_MODEL, SIMILARITY_THRESHOLD, TOKEN_LIMIT

with open("credentials.json", "r") as f:
    credentials = json.load(f)

if ENCODING_MODEL == "remote":
    openai.api_key = credentials["openai"]["api_key"]
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print(
            f"You are using {device}. This is much slower than using "
            "a CUDA-enabled GPU. If on Colab you can change this by "
            "clicking Runtime > Change runtime type > GPU."
        )

    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# get api key from app.pinecone.io
api_key = credentials["pinecone"]["api_key"]
# find your environment next to the api key in pinecone console
env = credentials["pinecone"]["environment"]

pinecone.init(api_key=api_key, environment=env)

if ENCODING_MODEL == "remote":
    index_name = "notion-documents-openai"
else:
    index_name = "notion-documents-local"

# connect to index
index = pinecone.GRPCIndex(index_name)
vector_count = index.describe_index_stats()["total_vector_count"]

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")


def retrieve(prompt, top_k: int = 10, context_padding: int = 0) -> str:
    # retrieve from Pinecone
    if ENCODING_MODEL == "remote":
        res = openai.Embedding.create(input=[prompt], engine="text-embedding-ada-002")
        xq = res["data"][0]["embedding"]
    else:
        xq = model.encode([prompt])[0]

    # get relevant contexts (including the questions)
    matches = index.query(xq, top_k=top_k, include_metadata=True)["matches"]
    contexts = []
    for match in matches:
        if match["score"] < SIMILARITY_THRESHOLD:
            continue
        text = match["metadata"]["text"]
        if context_padding <= 0:
            contexts.append(text)
            continue
        id = match["id"]
        texts = [r["metadata"]["text"] for r in get_rows_by_id(id, context_padding)]
        texts.insert(context_padding, text)
        contexts.append("\n\n".join(texts))

    if not contexts:
        raise ValueError("No relevant contexts found.")

    prompt_start = "Answer the question based on the context below.\n\n" + CONTEXT_PROMPT_NOTES + "\n\n" + "Context:\n"
    prompt_end = f"\n\nQuestion: {prompt}\nDetailed answer:"
    total_tokens = len(enc.encode(prompt_start + prompt_end))
    usable_contexts = []
    for context in contexts:
        total_tokens += len(enc.encode(context))
        if total_tokens < TOKEN_LIMIT:
            usable_contexts.append(context)
        else:
            break
    return prompt_start + "\n\n---\n\n".join(usable_contexts) + prompt_end


def get_rows_by_id(id: Union[str, int], padding: int = 0) -> list:
    """
    Get the rows before and after the row with the given id.

    Attributes:
        id: The id of the row to get the context for.
        padding: The number of rows to get before and after the row with the given id.

    Returns:
        A list of rows before and after the row with the given id.
    """
    if not padding:
        return []

    id = int(id)
    contexts = []
    for i in range(1, padding + 1):
        if id - i >= 0:
            res = index.query(id=str(id - i), top_k=1, include_metadata=True)
            contexts.extend(res["matches"])
        if id + i < vector_count:
            res = index.query(id=str(id + i), top_k=1, include_metadata=True)
            contexts.extend(res["matches"])
    return contexts
