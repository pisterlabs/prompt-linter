""" Module for the multiple visualization layer """
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

from langchain.chains import LLMChain
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.pydantic_v1 import Field as PyField

from fre_cohen import configuration
from fre_cohen.data_structure import (
    CompositeField,
    FieldsGraph,
    GraphsLayout,
    IndividualGraph,
)
from fre_cohen.llms import DEFAULT_RETRIES, LLMQualityEnum, build_llm_chain

logger = logging.getLogger(__name__)


class MultipleVisualizationLayer(ABC):
    """Abstract class for the multiple visualization layer"""

    def __init__(self, fields_graph: FieldsGraph):
        self._fields_graph = fields_graph

    @abstractmethod
    def get_layout(self) -> GraphsLayout:
        """Returns the data structure"""


class LayoutItem(BaseModel):
    """Item of the layout"""

    title: str = PyField("", description="Name of the layout")
    fields: list[str] = PyField([], description="Fields composing the layout")


class LayoutInfo(BaseModel):
    """Layout information"""

    titles: list[str] = PyField([], description="List of titles for the layouts")
    chart_descriptions: list[str] = PyField(
        [],
        description="List of chart descriptions, i.e. what kind of representation to use?",
    )
    independent_fields: list[list[str]] = PyField(
        [],
        description="List of the lists of independent variable fields for the layouts",
    )
    dependent_fields: list[list[str]] = PyField(
        [],
        description="List of the lists of dependent variable fields for the layouts",
    )


class LLMMultipleVisualizationLayer(MultipleVisualizationLayer):
    """LLM-based multiple visualization layer"""

    def __init__(self, config: configuration.Config, fields_graph: FieldsGraph):
        super().__init__(fields_graph)

        self._llm_layout = self._build_llm_chain_for_layout(config)

    def get_layout(self) -> GraphsLayout:
        """Computes a layout for the graph"""
        return self._generate_layout()

    def _get_node_index_by_name(self, name: str) -> int:
        """Returns the index of the node by name"""
        name = name.lower()
        for index, node in enumerate(self._fields_graph.nodes):
            if node.name.lower() == name:
                return index
        raise ValueError(f"Node with name {name} not found")

    def _get_node_by_name(self, name: str) -> CompositeField:
        """Returns the node by name"""
        name = name.lower()
        for node in self._fields_graph.nodes:
            if node.name.lower() == name:
                return node
        raise ValueError(f"Node with name {name} not found")

    def _generate_layout(self) -> GraphsLayout:
        """Generates the layout"""

        retries = 0
        exception_instructions: list[str] = []
        layout_info: Optional[LayoutInfo] = None

        while retries < DEFAULT_RETRIES:
            try:
                fields_summary = "\n".join(
                    [
                        f'{field.name}: "{field.description}"'
                        for field in self._fields_graph.nodes
                    ]
                )
                input_data = {
                    "all_fields_details": fields_summary,
                    "dependent_fields": "\n".join(
                        [
                            f"{self._fields_graph.nodes[edge.target].name} depends on {self._fields_graph.nodes[edge.source].name}"
                            for edge in self._fields_graph.edges
                        ]
                    ),
                    "exception_instructions": "\n".join(exception_instructions),
                }
                logger.debug("Layout LLM input: %s", input_data)
                layout_info = self._llm_layout.run(input_data)
                logger.debug("Layout LLM output: %s", layout_info)
                if not layout_info:
                    raise RuntimeError("Layout LLM failed to generate a layout")

                tuples = zip(
                    layout_info.titles,
                    layout_info.chart_descriptions,
                    layout_info.independent_fields,
                    layout_info.dependent_fields,
                    strict=True,
                )

                return GraphsLayout(
                    fields_graph=self._fields_graph,
                    graphs=[
                        IndividualGraph(
                            title=title,
                            chart_description=chart_description,
                            independent_variables=[
                                self._get_node_index_by_name(field_index)
                                for field_index in independent_fields
                            ],
                            dependent_variables=[
                                self._get_node_index_by_name(field_index)
                                for field_index in dependent_fields
                            ],
                        )
                        for title, chart_description, independent_fields, dependent_fields in tuples
                    ],
                )
            except Exception as e:
                logger.warning("Error in Vega LLM: %s", e)
                exception_instructions = [
                    f"Please correct your previous answer: {layout_info.json() if layout_info else None}",
                    f"Because it contains this error: {repr(e)}",
                ]
                retries += 1
        raise RuntimeError(f"Layout LLM failed after {DEFAULT_RETRIES} retries")

    def _build_llm_chain_for_layout(self, config: configuration.Config) -> LLMChain:
        """Builds a LLMChain to enrich a field"""
        return build_llm_chain(
            config,
            LayoutInfo,
            [
                SystemMessagePromptTemplate.from_template(
                    "Here are the fields composing our data set: {all_fields_details}"
                ),
                # SystemMessagePromptTemplate.from_template(
                #     "Here are the dependencies between the fields: {dependent_fields}"
                # ),
                SystemMessagePromptTemplate.from_template(
                    "Given a large number of fields, I want to split them into multiple visualizations with fewer fields for each. Can you suggest a way to group the fields based on their relationships or similarities, and then identify the independent and dependent variables for each visualization? What would be meaningful titles for each visualization, describing what is its purpose? How would you describe their representation (what type of visualization to use for representing them)?"
                ),
                SystemMessagePromptTemplate.from_template(
                    "You can reuse the same independent variable fields for multiple graphs. All the fields must be used at least once. In a graph, a variable cannot be both dependent and independent."
                ),
                SystemMessagePromptTemplate.from_template(
                    "{exception_instructions}",
                ),
            ],
            quality=LLMQualityEnum.ACCURACY,
        )
