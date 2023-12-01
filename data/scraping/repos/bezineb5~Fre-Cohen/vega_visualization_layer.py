"""
The visualization layer is responsible for generating the visualization of the fields graph.
The output is a valid Vega-lite specification.
"""
import json
import logging
from typing import Any, Optional, Sequence

import altair as alt
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    BaseMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.pydantic_v1 import BaseModel
from langchain.pydantic_v1 import Field as PyField
from langchain.pydantic_v1 import validator

from fre_cohen import configuration
from fre_cohen.data_structure import (
    FieldsGraph,
    GraphSpecifications,
    IndividualGraph,
    InstructionOrigin,
    RichField,
)
from fre_cohen.llms import DEFAULT_RETRIES, LLMQualityEnum, build_llm_chain
from fre_cohen.mapbox_style_visualization_layer import LLMMapboxStyleSpecifications
from fre_cohen.visualization_layer import IndividualVisualizationLayer

logger = logging.getLogger(__name__)


class VegaSpecification(BaseModel):
    """Vega-lite specification"""

    specifications: Any = PyField(
        None,
        description="Specifications of the graph, e.g. the vega-lite specification.",
    )
    is_map: bool = PyField(
        False,
        description="True if this is a map or a geographical visualization, false otherwise.",
    )

    @validator("specifications", pre=True, always=True)
    def check_specifications(cls, v):
        if v is None:
            raise ValueError("Vega-lite specifications cannot be None")

        # Try to parse the specifications using altair
        try:
            alt.Chart.from_dict(v)
        except Exception as e:
            raise ValueError(
                f"Vega-lite specifications are not valid. Can you correct them, please? Specifications: {json.dumps(v)}. Error: {e}"
            ) from e

        return v


class LLMIndividualVegaVisualizationLayer(IndividualVisualizationLayer):
    """Individual visualization layer that uses LLM to generate the Vega-lite specification."""

    def __init__(
        self,
        config: configuration.Config,
        data_source: str,  # TODO: deprecated, as it's overwritten at rendering time
        fields_graph: FieldsGraph,
        graph: IndividualGraph,
        previous_specifications: Optional[GraphSpecifications],
    ):
        super().__init__(fields_graph, graph, previous_specifications)

        self._data_source = data_source
        self._llm_layout = self._build_llm_chain_for_vega(
            config,
            self._has_human_instructions(graph),
            self._has_critic(graph),
            self._has_previous(previous_specifications),
        )
        self._mapbox_style_layer = LLMMapboxStyleSpecifications(
            config, fields_graph, graph, None
        )

    def _has_human_instructions(self, graph: IndividualGraph) -> bool:
        """Returns true if the graph has human instructions"""
        return any(
            [
                instruction.origin == InstructionOrigin.HUMAN
                for instruction in graph.instructions
            ]
        )

    def _has_critic(self, graph: IndividualGraph) -> bool:
        """Returns true if the graph has critic instructions"""
        return any(
            [
                instruction.origin == InstructionOrigin.CRITIC
                for instruction in graph.instructions
            ]
        )

    def _has_previous(
        self, previous_specifications: Optional[GraphSpecifications]
    ) -> bool:
        """Returns true if the graph has previous specifications"""
        return (
            previous_specifications is not None
            and previous_specifications.specifications is not None
        )

    # @retry_on_error
    def get_specifications(self) -> GraphSpecifications:
        """Returns the specifications of the graph"""

        # Run the LLM chain
        output = self._get_vega_lite_specifications()

        if output.is_map:
            map_style = self._get_map_style()
        else:
            map_style = None

        return GraphSpecifications(
            format_type="vega-lite",
            specifications=output.specifications,
            graph=self._graph,
            map_style=map_style,
        )

    def _get_map_style(self) -> Optional[str]:
        """Returns the map style"""
        gspec = self._mapbox_style_layer.get_specifications()
        return gspec.specifications

    def _summarize_fields(self, fields: Sequence[RichField]) -> str:
        """Summarizes the fields"""
        return "\n".join(
            [
                f'* "{field.field.name}": "{field.description}" with unit {field.unit} and with summary: "{field.field.summary}"'
                for field in fields
            ]
        )

    def _summarize_composite_fields(self, variable_indexes: Sequence[int]) -> str:
        """Summarizes the variables"""
        variables = [self._fields_graph.nodes[index] for index in variable_indexes]
        return "\n".join(
            [
                f'Fields related to {composite_field.description}:\n{self._summarize_fields(composite_field.columns)}\n'
                for composite_field in variables
            ]
        )

    def _get_vega_lite_specifications(self) -> VegaSpecification:
        """Returns the Vega-lite specifications"""

        retries = 0
        exception_instructions: list[str] = []

        while retries < DEFAULT_RETRIES:
            try:
                # Build the input data
                input_data = {
                    "data_source": self._data_source,
                    "title": self._graph.title,
                    "independent_variables_summary": self._summarize_composite_fields(
                        self._graph.independent_variables
                    ),
                    "dependent_variables_summary": self._summarize_composite_fields(
                        self._graph.dependent_variables
                    ),
                    "critic_advices": "\n".join(
                        [
                            instruction.instruction
                            for instruction in self._graph.instructions
                            if instruction.origin == InstructionOrigin.CRITIC
                        ]
                        + exception_instructions
                    ),
                    "human_instructions": "\n".join(
                        [
                            instruction.instruction
                            for instruction in self._graph.instructions
                            if instruction.origin == InstructionOrigin.HUMAN
                        ]
                    ),
                    "previous_specifications": self._previous_specifications.specifications
                    if self._previous_specifications is not None
                    else None,
                }
                logger.debug("Vega LLM input: %s", input_data)
                output: VegaSpecification = self._llm_layout.run(input_data)
                logger.debug("Vega LLM output: %s", output)
                return output
            except Exception as e:
                logger.warning("Error in Vega LLM: %s", e)
                exception_instructions = [repr(e)]
                retries += 1
        raise RuntimeError(f"Vega LLM failed after {DEFAULT_RETRIES} retries")

    def _contains_map(self, vega_spec: VegaSpecification) -> bool:
        """Returns true if the vega specification contains a map"""
        return "map" in vega_spec.specifications["mark"]

    def _build_llm_chain_for_vega(
        self,
        config: configuration.Config,
        has_human_instructions: bool,
        has_critic: bool,
        has_previous: bool,
    ) -> LLMChain:
        """Builds the LLM chain for vega-lite specification"""

        message_templates: list[BaseMessagePromptTemplate] = [
            SystemMessagePromptTemplate.from_template(
                "This is the title for the visualization: {title}"
            ),
            SystemMessagePromptTemplate.from_template(
                'This is the datasource path for the visualization: "{data_source}"'
            ),
            SystemMessagePromptTemplate.from_template(
                "These are the independent variables for the visualization:\n{independent_variables_summary}"
            ),
            SystemMessagePromptTemplate.from_template(
                "These are the dependent variables for the visualization:\n{dependent_variables_summary}"
            ),
            HumanMessagePromptTemplate.from_template(
                "What kind of visualization would be best suited for this data? Is this a geographical visualization?"
            ),
            HumanMessagePromptTemplate.from_template(
                "Given a set of data and a description of the visualization you want to create, can you generate a vega-lite visualization in JSON format that will produce the desired visualization? Please include the data source, the encoding of the data, and any necessary transformations or scales."
            ),
        ]

        if has_human_instructions:
            message_templates.append(
                HumanMessagePromptTemplate.from_template(
                    "Here are some instructions: {human_instructions}",
                ),
            )
        if has_critic:
            message_templates.append(
                SystemMessagePromptTemplate.from_template(
                    "Here are some feedback from a review: {critic_advices}",
                ),
            )
        if has_previous:
            message_templates.append(
                HumanMessagePromptTemplate.from_template(
                    "Can you adjust the visualization to address the feedback? Here is the previous visualization: {previous_specifications}",
                ),
            )

        llm_chain = build_llm_chain(
            config,
            VegaSpecification,
            message_templates,
            LLMQualityEnum.ACCURACY,
        )

        return llm_chain
