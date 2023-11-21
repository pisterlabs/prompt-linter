import shlex
from typing import Dict, List

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain_contrib.prompts import ChainedPromptTemplate
from pydantic import BaseModel

from zamm.actions.edit_file import EditFileChain


class TerminalChain(LLMChain, BaseModel):
    """Asks LLM for terminal input, and executes it"""

    terminal_chain: Chain

    @property
    def input_keys(self) -> List[str]:
        """Input keys to this chain."""
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        return ["command", "output"]

    def _validate_outputs(self, outputs: Dict[str, str]) -> None:
        if "file_path" in outputs:
            # edit file chain was invoked instead, validation should really be handled
            # there
            return
        return super()._validate_outputs(outputs)

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        assert isinstance(self.llm, BaseLLM)
        command = self.llm(self.prompt.format(**inputs), stop=["\n"]).strip()
        parsed_cmd = shlex.split(command)
        if len(parsed_cmd) == 2 and parsed_cmd[0] == "nano":
            assert isinstance(self.prompt, ChainedPromptTemplate)
            prefix = self.prompt.subprompts[0]
            edit_result = EditFileChain.for_file(llm=self.llm, prefix=prefix)(
                {
                    **inputs,
                    "file_path": parsed_cmd[1],
                }
            )
            return edit_result

        results = self.terminal_chain(command)
        results.pop("choice", None)
        return results
