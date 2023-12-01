"""
Batch embeddings with retries.

Modified from
https://github.com/openai/openai-cookbook/blob/main/examples/Get_embeddings.ipynb
"""

import openai
from openai.openai_object import OpenAIObject
from tenacity import retry, wait_random_exponential, stop_after_attempt

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_batch_embeddings(texts: list[str], model="text-embedding-ada-002") -> list[list[float]]:
    response = openai.Embedding.create(input=texts, model=model)
    for i, be in enumerate(response["data"]):
        assert i == be["index"]  # double check embeddings are in same order as input
    batch_embeddings = [e["embedding"] for e in response["data"]]
    return batch_embeddings


def get_one_embedding(text: str, model="text-embedding-ada-002") -> OpenAIObject:
    response = openai.Embedding.create(input=text, model=model)
    return response

if __name__ == "__main__":

    texts = [
        "Physics is the natural science of matter, involving the study of matter, its fundamental constituents, its motion and behavior through space and time, and the related entities of energy and force. Physics is one of the most fundamental scientific disciplines, with its main goal being to understand how the universe behaves. A scientist who specializes in the field of physics is called a physicist.",
        "Physics is one of the oldest academic disciplines and, through its inclusion of astronomy, perhaps the oldest. Over much of the past two millennia, physics, chemistry, biology, and certain branches of mathematics were a part of natural philosophy, but during the Scientific Revolution in the 17th century these",
    ]

    embeddings = get_batch_embeddings(texts)
    embedding_response = get_one_embedding(texts[0])
