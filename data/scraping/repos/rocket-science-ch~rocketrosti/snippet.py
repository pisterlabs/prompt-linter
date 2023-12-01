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

"""Snippet type that supports merging and optionally has an embedding."""

from __future__ import annotations

import hashlib
from functools import cached_property
from pathlib import Path
from typing import IO, Iterable, Sequence, cast

import attrs
import numpy as np
import orjson
from attrs import field
from attrs.setters import frozen, validate
from loguru import logger

from rrosti.llm_api import openai_api
from rrosti.utils.config import config
from rrosti.utils.misc import FloatArray


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_str(data: str) -> str:
    return _hash_bytes(data.encode("utf-8"))


def _embedding_cache_path() -> Path:
    return config.document_sync.data_gen_path / "embeddings.map"


# TODO: Some of this logic could be simplified, by e.g. having a transparent-ish cache
# that queries the API when it doesn't have an embedding.
class EmbeddingCache:
    _private_tag = object()
    _map: dict[str, FloatArray]
    _sorted: bool

    def _ensure_sorted(self) -> None:
        if not self._sorted:
            self._map = dict(sorted(self._map.items()))
            self._sorted = True

    def __init__(self, private_tag: object) -> None:
        """Internal constructor."""
        assert private_tag is self._private_tag, "EmbeddingCache should not be instantiated directly."
        self._map = {}
        self._sorted = True

    @classmethod
    def load(cls) -> EmbeddingCache:
        cache = cls(cls._private_tag)

        path = _embedding_cache_path()

        if not path.exists():
            logger.info("No embedding cache found, creating new one.")
            return cache

        logger.info("Loading embedding cache from {}...", path)

        hash_list: list[str] = []
        all_bytes: list[bytes] = []

        with open(path) as f:
            for line in f:
                hash, bytes_hex = line.strip().split("\t")
                hash_list.append(hash)
                all_bytes.append(bytes.fromhex(bytes_hex))

        embeddings = np.frombuffer(b"".join(all_bytes), dtype=np.float32).reshape(len(hash_list), -1)

        cache._map = dict(zip(hash_list, embeddings))
        cache._sorted = False

        assert all(len(k) == 64 for k in cache._map)
        assert len({len(v) for v in cache._map.values()}) == 1

        logger.info("Loaded a cache of {} embeddings.", len(cache._map))

        return cache

    def save(self) -> None:
        self._ensure_sorted()

        path = _embedding_cache_path()

        # Atomically replace
        fname_new = Path(str(path) + ".new")
        with open(fname_new, "w") as f:
            for hash, embedding in self._map.items():
                f.write(f"{hash}\t{embedding.tobytes().hex()}\n")

        fname_new.rename(path)

    def _assert_consistency(self, snippet: Snippet) -> None:
        if snippet._embedding is None:
            return
        cached = self._map.get(snippet.hash)
        if cached is None:
            return

        assert snippet._embedding.shape == cached.shape, (snippet._embedding.shape, cached.shape)
        if not np.all(snippet._embedding == cached):
            # Tolerate a small cosine distance between the two embeddings.
            cos_dist = 1 - np.dot(snippet._embedding, cached) / np.linalg.norm(snippet._embedding) / np.linalg.norm(
                cached
            )

            if cos_dist > 0.01:
                logger.error("Embedding cache is inconsistent with snippet.")
                logger.error("Snippet:\n{}", snippet.text)
                logger.error("Snippet embedding:\n{}", snippet._embedding)
                logger.error("Cached embedding:\n{}", cached)
                logger.error("Cosine distance: {}", cos_dist)
                assert False

            # Close enough. Just use the cached embedding.
            snippet._embedding = cached

    def _copy_to_snippet(self, snippet: Snippet) -> None:
        if snippet._embedding is None and snippet.hash in self._map:
            snippet._embedding = self._map[snippet.hash].copy()
            return

    def _copy_from_snippet(self, snippet: Snippet) -> None:
        if snippet._embedding is not None and snippet.hash not in self._map:
            self._map[snippet.hash] = snippet._embedding.copy()

    def sync_with_snippet(self, snippet: Snippet) -> None:
        """
        If the snippet has an embedding, add it to the cache.

        If the cache has an embedding for the snippet, add it to the snippet.
        """

        self._assert_consistency(snippet)
        self._copy_from_snippet(snippet)
        self._copy_to_snippet(snippet)

    def sync_with_snippets(self, snippets: Iterable[Snippet]) -> None:
        for snippet in snippets:
            self.sync_with_snippet(snippet)

    @classmethod
    def from_snippets(cls, snippets: Iterable[Snippet]) -> EmbeddingCache:
        cache = cls(cls._private_tag)
        cache.sync_with_snippets(snippets)
        return cache


# TODO: Split into Snippet and SnippetWithEmbedding
@attrs.define(slots=False)  # slots=False because we use cached_property
class Snippet:
    """
    A snippet type that supports merging. Can optionally have an embedding.

    Embeddings are dense vector representations in a lower-dimensional space
    that capture semantic meaning, allowing for operations like similarity checks.
    """

    source_filename: str = field(
        on_setattr=frozen, kw_only=True, validator=[attrs.validators.instance_of(str)]
    )  # typically filename or similar identifier
    start_offset: int = field(on_setattr=frozen, kw_only=True, validator=[attrs.validators.instance_of(int)])
    text: str = field(on_setattr=frozen, validator=[attrs.validators.instance_of(str)])
    _embedding: FloatArray | None = field(on_setattr=validate, default=None, repr=False, kw_only=True)
    page_start: int | None
    page_end: int | None

    @classmethod
    def _drop_duplicates(cls, snippets: Iterable[Snippet]) -> list[Snippet]:
        hashes_seen: dict[str, int] = {}
        out: list[Snippet] = []
        dup_count = 0

        for snippet in snippets:
            if snippet.hash not in hashes_seen:
                hashes_seen[snippet.hash] = len(out)
                out.append(snippet)
            else:
                # If this one has an embedding and the previous one not, add the embedding to the previous one
                if snippet._embedding is not None and out[hashes_seen[snippet.hash]]._embedding is None:
                    out[hashes_seen[snippet.hash]]._embedding = snippet._embedding
                dup_count += 1

        if dup_count:
            logger.warning("Dropped {} duplicate snippets.", dup_count)

        return out

    @_embedding.validator
    def _validate_embedding(self, _attribute: attrs.Attribute[Snippet], value: FloatArray | None) -> None:
        if value is None:
            return
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.float32
        assert value.ndim == 1
        assert value.shape[0] > 0

    def sync_ensure_embedding(self, openai_provider: openai_api.OpenAIApiProvider) -> None:
        if self._embedding is None:
            self._embedding = openai_provider.create_embedding(input=[self.text]).embeddings[0]

    @classmethod
    def ensure_embeddings(cls, openai_provider: openai_api.OpenAIApiProvider, snippets: Sequence[Snippet]) -> None:
        """
        Ensure that all snippets have embeddings.

        This method will query OpenAI's API for embeddings for all snippets that do not have them.
        """
        snippets_without_embeddings = [s for s in snippets if s._embedding is None]
        if not snippets_without_embeddings:
            return

        logger.info(f"Querying embeddings for {len(snippets_without_embeddings)} snippets")
        embeddings = openai_provider.create_embedding([s.text for s in snippets_without_embeddings]).embeddings
        for snippet, embedding in zip(snippets_without_embeddings, embeddings):
            snippet._embedding = embedding

    async def async_ensure_embedding(self, openai_provider: openai_api.OpenAIApiProvider) -> None:
        if self._embedding is None:
            self._embedding = (await openai_provider.acreate_embedding([self.text])).embeddings[0]

    @classmethod
    async def async_ensure_embeddings(
        cls, openai_provider: openai_api.OpenAIApiProvider, snippets: Sequence[Snippet]
    ) -> None:
        """
        Ensure that all snippets have embeddings.

        This method will query OpenAI's API for embeddings for all snippets that do not have them.
        """
        snippets_without_embeddings = [s for s in snippets if s._embedding is None]
        if not snippets_without_embeddings:
            return

        logger.info(f"Querying embeddings for {len(snippets_without_embeddings)} snippets")
        embeddings = (await openai_provider.acreate_embedding([s.text for s in snippets_without_embeddings])).embeddings
        for snippet, embedding in zip(snippets_without_embeddings, embeddings):
            snippet._embedding = embedding

    @property
    def has_embedding(self) -> bool:
        return self._embedding is not None

    def sync_get_embedding(self, openai_provider: openai_api.OpenAIApiProvider) -> FloatArray:
        """Get the embedding for this snippet, requesting it synchronously if necessary."""
        self.sync_ensure_embedding(openai_provider)
        assert self._embedding is not None
        return self._embedding

    async def async_get_embedding(self, openai_provider: openai_api.OpenAIApiProvider) -> FloatArray:
        """Get the embedding for this snippet, requesting it asynchronously if necessary."""
        await self.async_ensure_embedding(openai_provider)
        assert self._embedding is not None
        return self._embedding

    @cached_property
    def hash(self) -> str:
        """The hash of the snippet text."""
        return _hash_str(self.text)

    @property
    def end_offset(self) -> int:
        return self.start_offset + len(self.text)

    @property
    def length(self) -> int:
        return len(self.text)

    def try_merge(self, other: Snippet) -> Snippet | None:
        """Merge with a potentially overlapping later snippet, unless there is a gap."""
        if self.source_filename != other.source_filename:
            return None

        assert self.start_offset <= other.start_offset, "Snippets must be sorted"
        if other.start_offset > self.end_offset:
            return None
        added_text = other.text[self.end_offset - other.start_offset :]
        return Snippet(
            text=self.text + added_text,
            start_offset=self.start_offset,
            source_filename=self.source_filename,
            embedding=None,
            page_start=self.page_start,
            page_end=other.page_end,
        )

    @staticmethod
    def from_query(query: str) -> Snippet:
        return Snippet(
            text=query, start_offset=0, source_filename="$$query", embedding=None, page_start=None, page_end=None
        )

    @staticmethod
    def merge_list(snippets: Iterable[Snippet]) -> list[Snippet]:
        """
        Given a list of snippets, merge snippets that are next to each other or that overlap.

        Changes snippet order.
        """

        snippets = sorted(snippets, key=lambda s: (s.source_filename, s.start_offset))

        merged_snippets: list[Snippet] = []
        for snippet in snippets:
            if not merged_snippets:
                merged_snippets.append(snippet)
                continue

            last_snippet = merged_snippets[-1]
            merged_snippet = last_snippet.try_merge(snippet)
            if merged_snippet is not None:
                merged_snippets[-1] = merged_snippet
            else:
                merged_snippets.append(snippet)

        return merged_snippets

    @staticmethod
    def from_dict(data: dict[str, int | str | None]) -> Snippet:
        """
        Load a snippet from a dictionary.

        These snippets are required to have embeddings.
        """
        assert isinstance(data["start"], int)
        assert isinstance(data["doc"], str)
        assert isinstance(data["content"], str)
        assert isinstance(data["page_start"], int | None)
        assert isinstance(data["page_end"], int | None)
        if data["embedding"] is None:
            emb: FloatArray | None = None
        else:
            assert isinstance(data["embedding"], str)
            emb = np.frombuffer(bytes.fromhex(data["embedding"]), dtype=np.float32)
        return Snippet(
            source_filename=data["doc"],
            start_offset=data["start"],
            text=data["content"],
            embedding=emb,
            page_start=data["page_start"],  # type: ignore[arg-type]
            page_end=data["page_end"],  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, int | str | None]:
        """Convert a snippet to a dictionary."""
        return {
            "start": self.start_offset,
            "doc": self.source_filename,
            "content": self.text,
            "embedding": self._embedding.tobytes().hex() if self._embedding is not None else None,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }

    @staticmethod
    def consolidate_embeddings(snippets: Sequence[Snippet]) -> FloatArray:
        """
        Consolidate the embeddings of a list of snippets into a single numpy array, replacing
        the embeddings in the snippets with slices of the array.

        The snippets must have embeddings.

        The larger array is useful for computing distances between snippets, while the slices
        are useful for taking less memory than fully copying the embeddings.
        """

        assert all(s._embedding is not None for s in snippets), "All snippets must have embeddings"
        assert all(
            cast(FloatArray, s._embedding).shape == cast(FloatArray, snippets[0]._embedding).shape for s in snippets
        ), "All embeddings must have the same shape"

        embeddings = np.stack([cast(FloatArray, s._embedding) for s in snippets])
        for s, e in zip(snippets, embeddings):
            s._embedding = e
        return embeddings

    @staticmethod
    def load_from_jsonl(file_or_filename: str | Path | IO[bytes]) -> list[Snippet]:
        """Load snippets from a JSONL file."""

        logger.info("Loading snippets from {}", file_or_filename)

        def _do(file: IO[bytes]) -> list[Snippet]:
            return [Snippet.from_dict(orjson.loads(line)) for line in file]

        if isinstance(file_or_filename, (str, Path)):
            with open(file_or_filename, "rb") as f:
                snippets = _do(f)
        else:
            snippets = _do(file_or_filename)

        logger.info("Loaded {} snippets from {}", len(snippets), file_or_filename)
        return snippets
