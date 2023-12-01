from typing import Any, Dict

from langchain.llms.base import BaseLLM
from langchain.schema import AgentAction
from langchain_contrib.prompts import ChainedPromptTemplate, Templatable

from zamm.actions.base import Action
from zamm.agents.z_step import ZStepOutput

from .chain import NoteChain
from .prompt import NOTE_LOGGER, NOTE_PROMPT


class NoteOutput(ZStepOutput):
    @classmethod
    def from_chain_output(cls, output: Dict[str, Any]):
        return cls(
            decision=AgentAction(
                tool=output["choice"],
                tool_input=output["note"],
                log="dummy log",
            ),
            observation="dummy observation",
            logger_template=NOTE_LOGGER,
        )

    @property
    def note(self):
        return self.decision.tool_input

    @property
    def template_args(self) -> Dict[str, str]:
        """Construct the dict used to render this output"""
        return {"note": self.note}


class MakeNote(Action):
    @classmethod
    def default(cls, llm: BaseLLM, prefix: Templatable):
        return cls(
            name="Make a mental note to yourself",
            output_type=NoteOutput,
            chain=NoteChain(
                llm=llm,
                prompt=ChainedPromptTemplate([prefix, NOTE_PROMPT]),
            ),
        )
