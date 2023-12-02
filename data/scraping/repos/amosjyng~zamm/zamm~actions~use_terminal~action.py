from typing import Any, Dict, Optional

from langchain.chains.base import Chain
from langchain.llms.base import BaseLLM
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import AgentAction
from langchain_contrib.prompts import ChainedPromptTemplate, Templatable

from zamm.actions.base import Action
from zamm.actions.edit_file import EditFileOutput
from zamm.agents.step import StepOutput
from zamm.agents.z_step import ZStepOutput
from zamm.utils import safe_format

from .chain import TerminalChain
from .prompt import (
    TERMINAL_USAGE_PREFIX,
    TERMINAL_USAGE_SUFFIX,
    TerminalPromptTemplate,
    TerminalUsageLogger,
)


class TerminalOutput(ZStepOutput):
    @classmethod
    def from_chain_output(cls, output: Dict[str, Any]):
        if "file_path" in output:
            return EditFileOutput.from_chain_output(output)
        return cls(
            decision=AgentAction(
                tool=output["choice"],
                tool_input=output["command"],
                log="dummy log",
            ),
            observation=output["output"],
            logger_template=TerminalUsageLogger(),
        )

    def _log(
        self,
        condensed: bool,
        previous: Optional[StepOutput],
        next: Optional[StepOutput],
    ) -> str:
        template = self.logger_template
        if self.output == "":
            template = PromptTemplate(
                input_variables=["command"], template="$ {command}"
            )
        if previous is None or not isinstance(previous, TerminalOutput):
            template = ChainedPromptTemplate(
                joiner="\n", subprompts=[TERMINAL_USAGE_PREFIX, template]
            )
        if next is None or not isinstance(next, TerminalOutput):
            template = ChainedPromptTemplate(
                joiner="\n", subprompts=[template, TERMINAL_USAGE_SUFFIX]
            )
        return safe_format(template, self.template_args)

    @property
    def command(self):
        return self.decision.tool_input

    @property
    def output(self):
        return self.observation

    @property
    def template_args(self) -> Dict[str, str]:
        """Construct the dict used to render this output"""
        return {"command": self.command, "output": self.output}


class UseTerminal(Action):
    @classmethod
    def default(cls, llm: BaseLLM, prefix: Templatable, terminal_chain: Chain):
        return cls(
            name="Use the terminal (to run a command, not to edit a file)",
            output_type=TerminalOutput,
            chain=TerminalChain(
                llm=llm,
                prompt=TerminalPromptTemplate(prefix=prefix),
                terminal_chain=terminal_chain,
            ),
        )
