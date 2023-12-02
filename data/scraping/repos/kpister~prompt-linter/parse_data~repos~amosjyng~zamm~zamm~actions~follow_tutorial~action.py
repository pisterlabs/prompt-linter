from typing import Any, Callable, Dict

from langchain.agents.agent import AgentExecutor
from langchain.llms.base import BaseLLM
from langchain.schema import AgentAction
from langchain_contrib.prompts import ChainedPromptTemplate, Templatable

from zamm.actions.base import Action
from zamm.agents.z_step import ZStepOutput

from .chain import FollowTutorialChain
from .prompt import FOLLOW_TUTORIAL_LOGGER, FOLLOW_TUTORIAL_PROMPT


class FollowTutorialOutput(ZStepOutput):
    documentation_path: str
    documentation: str
    task: str
    output: str

    @classmethod
    def from_chain_output(cls, output: Dict[str, Any]):
        return cls(
            decision=AgentAction(
                tool=output["choice"],
                tool_input="dummy input",
                log="dummy log",
            ),
            observation="dummy observation",
            documentation_path=output["documentation_path"],
            documentation=output["documentation"],
            task=output["task"],
            output=output["output"],
            logger_template=FOLLOW_TUTORIAL_LOGGER,
        )

    @property
    def template_args(self) -> Dict[str, str]:
        """Construct the dict used to render this output"""
        return {"documentation_path": self.documentation_path, "task": self.task}


class FollowTutorial(Action):
    @classmethod
    def default(
        cls,
        llm: BaseLLM,
        prefix: Templatable,
        agent_creator: Callable[[], AgentExecutor],
    ):
        return cls(
            name="Follow instructions from a file",
            output_type=FollowTutorialOutput,
            chain=FollowTutorialChain(
                llm=llm,
                prompt=ChainedPromptTemplate([prefix, FOLLOW_TUTORIAL_PROMPT]),
                agent_creator=agent_creator,
            ),
        )
