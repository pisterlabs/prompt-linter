import itertools
import re

import bleach
import cohere
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from tiktoken import get_encoding

tokenizer = get_encoding("gpt2")


def clean_text(text: str):
    new_text = re.sub(r"\n+", "\n", text)
    new_text = re.sub(r"\s+", " ", new_text)
    new_text = new_text.encode("utf-8", "ignore").decode("utf-8")
    return new_text.strip()


def get_full_path(section: str, path: str) -> str:
    new_section = section.split(">")[-1].lower().replace(" ", "-")
    new_section = re.sub(r"[^a-zA-Z0-9-]", "", new_section)
    return path + "#" + new_section


def get_embeddings(texts: list, cohere_client: cohere.Client) -> list:
    response = cohere_client.embed(
        model="large",
        texts=texts,
    )
    return response.embeddings


def get_similar_docs(
    query: str,
    qdrant_client: QdrantClient,
    cohere_client: cohere.Client,
    collection_name: str,
    limit: int = 30,
) -> list:
    query_vector = get_embeddings([query], cohere_client=cohere_client)[0]
    hits = qdrant_client.search(
        collection_name=collection_name,
        query_vector=np.array(query_vector).tolist(),
        append_payload=True,
        limit=limit,
    )
    return hits


def rerank_docs(
    hits: list,
    query: str,
    model: CrossEncoder,
    limit: int = 5,
) -> list:
    payloads = [
        {
            "text": clean_text(hit.payload["text"]),
            "title": clean_text(hit.payload["title"].replace('"', "").replace("'", ""))
            if hit.payload["title"]
            else "",
            "path": clean_text(hit.payload["path"]),
            "full_path": get_full_path(hit.payload["section"], hit.payload["path"]),
            "section": hit.payload["section"].split(">")[-1],
        }
        for hit in hits
    ]
    sentence_combinations = [[query, p["text"]] for p in payloads]
    similarity_scores = model.predict(sentence_combinations)
    sim_scores_argsort = reversed(np.argsort(similarity_scores))
    return [payloads[idx] for idx in itertools.islice(sim_scores_argsort, limit)]


def build_prompt(query: str, references: list) -> str:
    prompt_prefix = [
        "You're a helpful assistant. Given the following extracted parts of Gitlab's employees handbook, provide a detailed answer based on these references. If you don't know the answer, say you couldn't find any relevant sources."
        "\n\n",
        "References:",
    ]

    prompt_suffix = [
        f"\n\nQuestion: {query}",
        "\nAnswer:",
    ]

    curr_len = len(tokenizer.encode("".join(prompt_prefix + prompt_suffix)))

    prompt_references = ""
    candidate_reference = ""
    for i, reference in enumerate(references, start=1):
        candidate_reference += (
            f"\n[{i}] {clean_text(reference['title'])}: {clean_text(reference['text'])}"
        )

        if curr_len + len(tokenizer.encode(candidate_reference)) > 1900:
            break

        prompt_references += candidate_reference
        curr_len += len(tokenizer.encode(candidate_reference))

    return "".join(prompt_prefix) + prompt_references + "".join(prompt_suffix)


def get_response(prompt: str, cohere_client: cohere.Client):
    response = cohere_client.generate(
        model="command-xlarge-nightly",
        prompt=prompt,
        max_tokens=200,
        temperature=0,
    )
    return response.generations[0].text


def clean_question(question: str):
    clean_question = bleach.clean(question)
    if len(clean_question) < 180:
        return clean_question
    raise ValueError("Question must be shorter than 180 characters.")
