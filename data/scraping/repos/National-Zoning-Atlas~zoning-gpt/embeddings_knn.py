from typing import Generator

import numpy as np
from openai.embeddings_utils import get_embedding

from ..types import District, PageSearchOutput
from .base import Searcher
from .utils import expand_term, fill_to_token_length, get_lookup_tables


class EmbeddingsKNNSearcher(Searcher):
    def __init__(self, k: int, label: str = ""):
        assert k > 0, "`k` must be >0 to use KNN search"

        self.k = k
        self.ds, self.df = get_lookup_tables()
        self.label = label

    def search(
        self, town: str, district: District, term: str
    ) -> Generator[PageSearchOutput, None, None]:
        query = next(expand_term(term)) + district.full_name + district.short_name
        query_embedding = np.array(get_embedding(query, "text-embedding-ada-002"))

        filtered_ds = self.ds.filter(lambda x: x["Town"] == town)
        filtered_ds.add_faiss_index("embeddings")

        result = filtered_ds.get_nearest_examples(
            "embeddings", query_embedding, self.k
        )
        for page, score in zip(result.examples["Page"], result.scores):
            yield PageSearchOutput(
                text=fill_to_token_length(page, self.df.loc[town], 2000),
                page_number=page,
                score=score,
                log={"label": self.label},
                highlight=[],
                query=query,
            )
