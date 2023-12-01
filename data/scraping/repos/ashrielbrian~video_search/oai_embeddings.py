import logging
from typing import List, Iterator

import openai
from tenacity import retry, retry_if_exception, wait_exponential, stop_after_attempt

from models import Segment
from utils import batch_segments


@retry(
    retry=retry_if_exception(openai.OpenAIError),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(4),
)
def get_multi_embeddings(
    texts: List[str], model="text-embedding-ada-002"
) -> List[List[float]]:
    texts = [text.replace("\n", " ") for text in texts]
    return [
        data["embedding"]
        for data in openai.Embedding.create(input=texts, model=model)["data"]
    ]


def generate_embeddings_batch(
    segments: List[Segment], batch_size: int = 50
) -> Iterator[List[Segment]]:
    for batch in batch_segments(segments, batch_size):
        try:
            batch_text = [row.text for row in batch]
            embs = get_multi_embeddings(batch_text)

            # assigning emb to the segment takes up a lof of memory.
            # consider commenting this out, and simply yielding embs
            for i, segment in enumerate(batch):
                segment.emb = embs[i]

            yield batch
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            print(f"Problematic batch: {batch}")
            raise e
