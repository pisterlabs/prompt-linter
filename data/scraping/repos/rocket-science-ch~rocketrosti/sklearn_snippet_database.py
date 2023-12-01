# Copyright (c) 2023 Rocket Science AG, Switzerland

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""An in-memory snippet vector database using sklearn's NearestNeighbors."""

from __future__ import annotations

import asyncio
from typing import Iterable

from loguru import logger
from overrides import override
from sklearn.neighbors import NearestNeighbors  # type: ignore[import]

from rrosti.llm_api import openai_api
from rrosti.snippets.abstract_snippet_database import AbstractSnippetDatabase
from rrosti.snippets.snippet import Snippet
from rrosti.utils.misc import FloatArray


class SklearnSnippetDatabase(AbstractSnippetDatabase):
    """An in-memory snippet vector database using sklearn's NearestNeighbors."""

    _nbrs: NearestNeighbors
    _snippets: list[Snippet]
    _embeddings: FloatArray
    _id_to_index_map: dict[str, int]

    def __init__(self, snippets: list[Snippet]) -> None:
        self._snippets = Snippet._drop_duplicates(snippets)
        self._id_to_index_map = {snippet.hash: i for i, snippet in enumerate(self._snippets)}
        self._embeddings = Snippet.consolidate_embeddings(self._snippets)
        self._nbrs = NearestNeighbors(n_neighbors=1, algorithm="brute", metric="cosine", n_jobs=-1).fit(
            self._embeddings
        )
        logger.info(f"Initialized SklearnSnippetDatabase with {len(self._snippets)} snippets")

    @override
    def has_id(self, id: str) -> bool:
        return id in self._id_to_index_map

    @override
    def __contains__(self, snippet: Snippet) -> bool:
        return snippet.hash in self._id_to_index_map

    @override
    def get_by_id(self, id: str) -> Snippet | None:
        if id not in self._id_to_index_map:
            return None
        return self._snippets[self._id_to_index_map[id]]

    @override
    def add_snippet(self, snippet: Snippet) -> Snippet:
        raise NotImplementedError("TODO: implement this")

    @override
    def add_snippets(self, snippets: Iterable[Snippet]) -> None:
        raise NotImplementedError("TODO: implement this")

    @override
    async def find_nearest_raw(
        self, openai_provider: openai_api.OpenAIApiProvider, query: Snippet, n_results: int
    ) -> list[Snippet]:
        emb = (await query.async_get_embedding(openai_provider)).reshape(1, -1)
        logger.info("Finding {} neighbors...", n_results)
        _, indices = await asyncio.to_thread(self._nbrs.kneighbors, emb, n_results)
        logger.info("Found neighbors")
        return [self._snippets[i] for i in indices[0]]
