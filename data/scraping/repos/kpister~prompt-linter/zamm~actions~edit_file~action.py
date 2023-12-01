from typing import Any, Dict, Optional

from langchain.llms.base import BaseLLM
from langchain.schema import AgentAction
from langchain_contrib.prompts import Templatable

from zamm.actions.base import Action
from zamm.agents.z_step import StepOutput, ZStepOutput
from zamm.utils import safe_format

from .chain import EditFileChain
from .prompt import CONDENSED_LOGGER, EDIT_LOGGER, NEW_FILE_LOGGER


class EditFileOutput(ZStepOutput):
    file_exists: bool
    old_contents: Optional[str]

    @classmethod
    def from_chain_output(cls, output: Dict[str, Any]):
        file_exists = output["file_exists"] == "True"
        if file_exists:
            old_contents = output["old_contents"]
        else:
            old_contents = None
        return cls(
            decision=AgentAction(
                tool=output["choice"],
                tool_input=output["file_path"],
                log="dummy log",
            ),
            observation=output["new_contents"],
            old_contents=old_contents,
            file_exists=file_exists,
            logger_template=EDIT_LOGGER,
        )

    @property
    def file_path(self):
        return self.decision.tool_input

    @property
    def new_contents(self):
        return self.observation

    @property
    def template_args(self) -> Dict[str, str]:
        """Construct the dict used to render this output"""
        args = {
            "file_path": self.file_path,
            "file_exists": self.file_exists,
            "new_contents": self.new_contents,
        }
        if self.file_exists:
            assert self.old_contents is not None
            args["old_contents"] = self.old_contents
        return args

    def _log(
        self,
        condensed: bool,
        previous: Optional[StepOutput],
        next: Optional[StepOutput],
    ) -> str:
        if condensed:
            return safe_format(CONDENSED_LOGGER, self.template_args)
        elif self.file_exists:
            return safe_format(EDIT_LOGGER, self.template_args)
        else:
            return safe_format(NEW_FILE_LOGGER, self.template_args)


class EditFile(Action):
    @classmethod
    def default(cls, llm: BaseLLM, prefix: Templatable):
        return cls(
            name="Edit a file",
            output_type=EditFileOutput,
            chain=EditFileChain.default(
                llm=llm,
                prefix=prefix,
            ),
        )
