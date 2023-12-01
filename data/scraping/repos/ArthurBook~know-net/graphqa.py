"""
Adapted: https://github.com/hwchase17/langchain/blob/master/langchain/chains/graph_qa/base.py
to query vectors
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, NamedTuple
import networkx as nx
import itertools
import loguru
from langchain.chains import GraphQAChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.graphs.networkx_graph import get_entities
from know_net.graph_building import LLMGraphBuilder
from know_net import graph_building
from know_net.graph_building import Entity

logger = loguru.logger


class EntityKnowledge(NamedTuple):
    triplet_string: str
    references: List[str]


class VecGraphQAChain(GraphQAChain):
    """Chain for question-answering against a graph."""

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Extract entities, look up info and answer question."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]

        entity_string = self.entity_extraction_chain.run(question)

        _run_manager.on_text("Entities Extracted:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            entity_string, color="green", end="\n", verbose=self.verbose
        )
        entities = get_entities(entity_string)
        logger.info("found entities: {}", entities)
        context = ""
        all_triplets: List[EntityKnowledge] = []
        for entity in entities:
            all_triplets.extend(get_entity_knowledge(self.graph, entity))
            # all_triplets.extend(self.graph.get_entity_knowledge(entity))
        context = "\n".join(t.triplet_string for t in all_triplets)
        _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
        _run_manager.on_text(context, color="green", end="\n", verbose=self.verbose)
        result = self.qa_chain(
            {"question": question, "context": context},
            callbacks=_run_manager.get_child(),
        )

        references = itertools.chain.from_iterable(t.references for t in all_triplets)
        urls = "\n".join(set(references))
        return {self.output_key: result[self.qa_chain.output_key], "references": urls}


def get_entity_knowledge(
    graph: LLMGraphBuilder, entity_str: str
) -> List[EntityKnowledge]:
    triplets: List[EntityKnowledge] = []
    results = graph.vectorstore.similarity_search_with_relevance_scores(entity_str)
    for doc, _ in results:
        entity = graph.doc_to_entity[doc.page_content]
        logger.info("entity:{}", entity)
        trip_str = str(get_entity_triples(graph.graph, entity))
        references = graph.graph.nodes[entity][graph_building.SOURCE_ATTR]
        logger.info("trip str: {}", trip_str)
        logger.info("urls: {}", set(references))
        triplets.append(EntityKnowledge(triplet_string=trip_str, references=references))
    return triplets


def get_entity_triples(graph: nx.Graph, entity: Entity, depth: int = 1) -> List[str]:
    """Get information about an entity."""

    if not graph.has_node(entity):
        return []

    results: List[str] = []
    for src, sink in nx.dfs_edges(graph, entity, depth_limit=depth):
        relation = graph[src][sink]["label"]
        results.append(f"({src.name}, {relation}, {sink.name})")
    return results
