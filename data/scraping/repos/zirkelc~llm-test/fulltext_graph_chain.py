from __future__ import annotations

from typing import Any, Dict, List

from langchain.chains.base import Chain
from langchain.graphs import Neo4jGraph
from pydantic import Field


class Neo4jFulltextGraphChain(Chain):
    """Chain for keyword question-answering against a graph."""

    graph: Neo4jGraph = Field(exclude=True)
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    fulltext_search: str = """
    CALL db.index.fulltext.queryNodes("fulltext_product", $query)
    YIELD node, score
    RETURN node.`$match` AS option, labels(node)[0] AS type LIMIT 3
    """

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.
        :meta private:
        """
        _output_keys = [self.output_key]
        return _output_keys

    def _call(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Extract entities, look up info and answer question."""
        question = inputs[self.input_key]
        context = self.graph.query(self.fulltext_search, {"query": question})
        return {self.output_key: context}
