import time
import warnings
from typing import Union

import cohere.error
import pandas as pd
import topically

from src.utils import get_logger

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = get_logger("topically")


class TopicallyClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.app = topically.Topically(api_key)

    @staticmethod
    def _chunk_input(
        texts: list[str], clusters: list[int], max_chunk_size: int = 100
    ) -> list[tuple[list[str], list[int]]]:

        df = pd.DataFrame({"text": texts, "cluster_label": clusters})

        chunks: list[pd.DataFrame] = []

        current_chunk_size = 0
        current_chunks: list[pd.DataFrame] = []
        for label, group in df.groupby(by=["cluster_label"]):
            if current_chunk_size < max_chunk_size:
                current_chunks.append(group)
                current_chunk_size += 1
            else:
                chunks.append(pd.concat(current_chunks))
                current_chunk_size = 1
                current_chunks = [group]
        if len(current_chunks) > 0:
            chunks.append(pd.concat(current_chunks))

        return [(df["text"].tolist(), df["cluster_label"].tolist()) for df in chunks]

    def name_topics(
        self,
        texts: list[str],
        clusters: list[int],
        max_chunk_size: int = 100,
        num_generations: int = 5,
        num_sample_texts: int = 20,
    ) -> Union[None, dict[int, str]]:

        chunks: list[tuple[list[str], list[int]]] = self._chunk_input(
            texts, clusters, max_chunk_size=max_chunk_size
        )
        logger.info(
            f"TopicallyClient will work with {len(chunks)} chunks of data, "
            f"each taking processing time + 60 seconds to complete."
        )
        topic_names: dict[int, str] = {}
        try:
            for index, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {index + 1} of {len(chunks)}")
                _, _topic_names = self.app.name_topics(
                    chunk,
                    num_generations=num_generations,
                    num_sample_texts=num_sample_texts,
                )
                topic_names.update(_topic_names)
                if index + 1 < len(chunks):
                    logger.info(
                        "Working with chunked input; will sleep 60 seconds to avoid throttling by Cohere."
                    )
                    time.sleep(60)
            return topic_names
        except cohere.error.CohereError as err:
            logger.error(f"Could not fulfil Cohere request: {err}")
            return None
