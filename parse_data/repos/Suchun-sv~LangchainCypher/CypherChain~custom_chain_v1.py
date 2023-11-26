"""Question answering over a graph."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import Field

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import CYPHER_QA_PROMPT
from langchain.chains.llm import LLMChain
from langchain.graphs.neo4j_graph import Neo4jGraph
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.prompts.prompt import PromptTemplate
Cases="""Right Cases:
querys: 列举出鲁迅的一个别名可以吗？ answer:match (:ENTITY{name:'鲁迅'})<--(h)-[:Relationship{name:'别名'}]->(q) return distinct q.name limit 1
querys: 我们常用的301SH不锈钢带的硬度公差是多少，你知道吗？ answers:match(p:ENTITY{name:'301SH不锈钢带'})-[:Relationship{name:'硬度公差'}]-> (q) return q.name
Wrong Cases:
querys: 12344加油这首歌真好听，你知道歌曲原唱是谁吗？ answers: MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) WHERE m.name = '12345加油' RETURN a.name LIMIT 30
querys: 七宗梦是什么时候上映的？ answers: MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) WHERE m.name = '七宗梦' RETURN a.name LIMIT 30"""


INTERMEDIATE_STEPS_KEY = "intermediate_steps"

# Use only the provided relationship types and properties in the schema.
# Do not use any other relationship types or properties that are not provided.

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Cases:
{cases}
Schema:
{schema}
Instructions:
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Do not generate a statement that query all the nodes or edges: MATCH (n:ENTITY)-[:Tag]->(m:ENTITY) RETURN n.name

The question is:
{question}"""
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question", "cases"], template=CYPHER_GENERATION_TEMPLATE
)


def extract_cypher(text: str) -> str:
    """Extract Cypher code from a text.

    Args:
        text: Text to extract Cypher code from.

    Returns:
        Cypher code extracted from the text.
    """
    # The pattern to find Cypher code enclosed in triple backticks
    pattern = r"```(.*?)```"

    # Find all matches in the input text
    matches = re.findall(pattern, text, re.DOTALL)

    return matches[0] if matches else text


class GraphCypherQAChain(Chain):
    """Chain for question-answering against a graph by generating Cypher statements."""

    graph: Neo4jGraph = Field(exclude=True)
    cypher_generation_chain: LLMChain
    qa_chain: LLMChain
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    top_k: int = 10
    """Number of results to return from the query"""
    return_intermediate_steps: bool = False
    """Whether or not to return the intermediate steps along with the final answer."""
    return_direct: bool = False
    """Whether or not to return the result of querying the graph directly."""

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

    @property
    def _chain_type(self) -> str:
        return "graph_cypher_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        qa_prompt: BasePromptTemplate = CYPHER_QA_PROMPT,
        cypher_prompt: BasePromptTemplate = CYPHER_GENERATION_PROMPT,
        **kwargs: Any,
    ) -> GraphCypherQAChain:
        """Initialize from LLM."""
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
        cypher_generation_chain = LLMChain(llm=llm, prompt=cypher_prompt)

        return cls(
            qa_chain=qa_chain,
            cypher_generation_chain=cypher_generation_chain,
            **kwargs,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Generate Cypher statement, use it to look up in db and answer question."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        question = inputs[self.input_key]

        intermediate_steps: List = []

        generated_cypher = self.cypher_generation_chain.run(
            {"question": question, "schema": self.graph.get_schema, "cases": Cases}, callbacks=callbacks
        )

        # Extract Cypher code if it is wrapped in backticks
        generated_cypher = extract_cypher(generated_cypher)

        if "limit" not in generated_cypher.lower():
            generated_cypher += " LIMIT 30"

        _run_manager.on_text("Generated Cypher:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            generated_cypher, color="green", end="\n", verbose=self.verbose
        )

        intermediate_steps.append({"query": generated_cypher})

        # Retrieve and limit the number of results
        context = self.graph.query(generated_cypher)[: self.top_k]
        # intermediate_steps['context'] = context
        intermediate_steps.append({"context": context})

        final_result = context

        chain_result: Dict[str, Any] = {self.output_key: final_result}
        if self.return_intermediate_steps:
            chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps

        return chain_result
