from typing import Callable, Dict, List

from langchain.agents.agent import AgentExecutor
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate
from pydantic import BaseModel

from zamm.chains.general.get_dict import GetDictChain, VariableConfig
from zamm.utils import read_documentation

from .prompt import FOLLOW_TUTORIAL_PROMPT


class FollowTutorialChain(LLMChain, BaseModel):
    """Asks LLM for which tutorial to follow."""

    agent_creator: Callable[[], AgentExecutor]
    prompt: BasePromptTemplate = FOLLOW_TUTORIAL_PROMPT

    @property
    def input_keys(self) -> List[str]:
        """Input keys to this chain."""
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        return ["documentation_path", "documentation", "task", "output"]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        prefix = self.prompt.format(**inputs)
        assert isinstance(self.llm, BaseLLM)
        tutorial_dict = GetDictChain(
            llm=self.llm,
            prefix=prefix,
            variable_configs=[
                VariableConfig.for_string(
                    output_key="documentation_path",
                    display="Path to the instructions file",
                    enclosure="`",
                ),
                VariableConfig(
                    output_key="task",
                    display="Task",
                ),
            ],
        )({})
        documentation = read_documentation(tutorial_dict["documentation_path"])

        employee = self.agent_creator()
        args = {
            "documentation": documentation,
            "task": tutorial_dict["task"],
        }
        result = employee(args)
        return {**tutorial_dict, **args, **result}
