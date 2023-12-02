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

"""
Base class for vector databases of snippets.

A snippet database is a collection of snippets that can be searched by vector similarity.
This is used to find the most relevant snippets for a query.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Iterable

from loguru import logger

from rrosti.llm_api import openai_api
from rrosti.snippets.snippet import Snippet
from rrosti.utils.token_count import token_count


def _count_tokens(question: str, snippets: list[str]) -> int:
    """Returns the token count of the query that would be sent to the LLM."""
    # FIXME: This estimates the token count in some way by procuding something that looks a bit like a prompt.
    # It used to be sillier, using an old jinja template that was not used anywhere else.
    # Now it tries to emulate that somewhat for compatibility.
    sysmsg = "\n\n".join(f"## Extract #{i+1}:\n\n{snippet}" for i, snippet in enumerate(snippets))
    return token_count(sysmsg) + token_count(question) + 80


class AbstractSnippetDatabase(ABC):
    """
    Base class for vector databases of snippets.

    A snippet database is a collection of snippets that can be searched by vector similarity.
    This is used to find the most relevant snippets for a query.
    """

    @abstractmethod
    def has_id(self, id: str) -> bool:
        """Check if the id (text hash) is in the database."""

    @abstractmethod
    def __contains__(self, snippet: Snippet) -> bool:
        """Check if the snippet is in the database."""

    @abstractmethod
    def get_by_id(self, id: str) -> Snippet | None:
        """Get a snippet by its id (text hash)."""

    @abstractmethod
    def add_snippet(self, snippet: Snippet) -> Snippet:
        """
        Add a snippet to the database.

        Returns the snippet that was added, which may be different from the input snippet
        if the snippet was already in the database.
        """

    @abstractmethod
    def add_snippets(self, snippets: Iterable[Snippet]) -> None:
        """Add multiple snippets to the database. Upserts."""

    @abstractmethod
    async def find_nearest_raw(
        self, openai_provider: openai_api.OpenAIApiProvider, query: Snippet, n_results: int
    ) -> list[Snippet]:
        """
        Find the nearest raw (unmerged) snippets to the query.

        Returns a list of snippets sorted by distance.
        """

    async def find_nearest_merged(
        self, openai_provider: openai_api.OpenAIApiProvider, query: Snippet, max_tokens: int, n_merge_candidates: int
    ) -> list[Snippet]:
        """
        query: The query to find snippets for. Used to determine the total length of snippets.
        max_tokens: The maximum number of tokens to return.
        n_merge_candidates: How many raw snippets to fetch and consider for merging.

        Returns (snippets: (1..M,)), where M <= n_merge_candidates.

        Distances are not returned because they are not well defined for merged snippets.
        """

        await query.async_ensure_embedding(openai_provider)

        candidates = await self.find_nearest_raw(openai_provider, query, n_merge_candidates)
        while True:
            merged = Snippet.merge_list(candidates)
            logger.debug("{} candidates merged into {} snippets", len(candidates), len(merged))
            # FIXME: This looks fishy: I think only qa_token_count is used from that module, and it relies on a prompt
            # template that we don't use otherwise.
            token_count = await asyncio.to_thread(_count_tokens, query.text, [m.text for m in merged])
            if token_count > max_tokens:
                # too long
                logger.debug("{} snippets is too long: {} > {}", len(candidates), token_count, max_tokens)
                candidates = candidates[:-1]
            else:
                logger.info("Going to query with {} snippets, {} after merging", len(candidates), len(merged))
                logger.debug("Tokens: {} <= {}", token_count, max_tokens)
                return merged
