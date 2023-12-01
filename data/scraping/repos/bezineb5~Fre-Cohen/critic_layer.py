"""The critic layer is responsible for generating the critic of the fields graph.
"""

import base64
import logging
import pathlib
import tempfile
from abc import ABC, abstractmethod
from typing import Callable, Optional

from langchain.output_parsers import MarkdownListOutputParser
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.pydantic_v1 import Field as PyField
from langchain.schema.messages import HumanMessage

from fre_cohen import configuration
from fre_cohen.data_structure import GraphSpecifications, Instruction, InstructionOrigin
from fre_cohen.llms import LLMQualityEnum, get_llm, retry_on_error
from fre_cohen.rendering.visualization_rendering import render_graph

logger = logging.getLogger(__name__)


class Critic(BaseModel):
    """Critic of the visualization"""

    advices: list[str] = PyField(
        "", description="List of comments to improve the visualization"
    )


class VisualizationCriticLayer(ABC):
    """Abstract class for the visualization critic layer"""

    def __init__(self, graph: GraphSpecifications):
        self._graph = graph

    @abstractmethod
    def enrich_with_critic_advices(self) -> GraphSpecifications:
        """Returns the description of the graph augemnted with critics."""


class LLMVisualizationCriticLayer(VisualizationCriticLayer):
    """LLM-based visualization critic layer"""

    def __init__(
        self,
        config: configuration.Config,
        graph: GraphSpecifications,
        data_source: pathlib.Path,
        mbtiles_path: Optional[pathlib.Path],
        fonts_path: Optional[pathlib.Path],
    ):
        super().__init__(graph)

        self._data_source = data_source
        self._mbtiles_path = mbtiles_path
        self._fonts_path = fonts_path

        self._llm_critic = self._build_llm_for_critic(config)

    def enrich_with_critic_advices(self) -> GraphSpecifications:
        """Computes a critic for the graph"""

        # First, render the visualization in PNG format
        with tempfile.NamedTemporaryFile(suffix=".png") as temp:
            output_file = pathlib.Path(temp.name)
            render_graph(
                self._graph,
                self._data_source,
                output_file,
                mbtiles_path=self._mbtiles_path,
                fonts_path=self._fonts_path,
            )
            logger.info("Rendered visualization to: %s", output_file)

            # Encode the image in base64
            file_bytes = output_file.read_bytes()

            # Encode the bytes to base64
            encoded_image = base64.b64encode(file_bytes).decode("utf-8")

            # Ask the LLM to critic it
            full_desc = (
                f"{self._graph.graph.title}: {self._graph.graph.chart_description}"
            )
            critic_response = self._llm_critic(encoded_image, full_desc)

            # Integrate the critic response into the graph
            current_instructions: list[Instruction] = list(
                self._graph.graph.instructions or []
            )
            for advice in critic_response.advices:
                current_instructions.append(
                    Instruction(
                        instruction=advice,
                        origin=InstructionOrigin.CRITIC,
                    )
                )
            with_instructions = self._graph.graph.model_copy(
                update={"instructions": current_instructions}
            )
            return self._graph.model_copy(update={"graph": with_instructions})

    def _build_llm_for_critic(
        self, config: configuration.Config
    ) -> Callable[[str, str], Critic]:
        vision_llm = get_llm(config, quality=LLMQualityEnum.VISION)

        output_parser = MarkdownListOutputParser()
        output_format_message = SystemMessagePromptTemplate.from_template(
            "{format_instructions}"
        ).format(format_instructions=output_parser.get_format_instructions())

        @retry_on_error
        def apply_llm(base64_image: str, description: str) -> Critic:
            """Applies the LLM to the image"""

            output_message = vision_llm.invoke(
                [
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": f"This visualization represents the following: {description}. Do you think it's clear, actionable, and easy to understand? If not, please provide some advices on how to improve it.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ]
                    ),
                    output_format_message,
                ]
            )

            output = output_parser.invoke(output_message)
            logger.debug("LLM output: %s", output)
            return Critic(advices=output)

        return apply_llm
