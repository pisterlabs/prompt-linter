from __future__ import annotations

from typing import Callable, List, Optional

from langchain_contrib.llms import Human
from langchain_contrib.llms.testing import FakeLLM
from vcr.cassette import Cassette
from vcr_langchain.patch import GenericPatch, add_patchers


class ZHuman(Human, FakeLLM):
    def _uncached_queries(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return Human._call(self, prompt, stop)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Run the LLM on the given prompt and input."""
        return FakeLLM._call(self, prompt, stop)


class ZHumanPatch(GenericPatch):
    """Patch for the ZHuman class."""

    def __init__(self, cassette: Cassette) -> None:
        """Initialize patch."""
        super().__init__(cassette, ZHuman, "_call")

    def get_same_signature_override(self) -> Callable:
        """Obtain same-signature override for ZHuman._call."""

        def _call(
            og_self: ZHuman, prompt: str, stop: Optional[List[str]] = None
        ) -> str:
            return self.generic_override(og_self, prompt=prompt, stop=stop)

        return _call


add_patchers(ZHumanPatch)
