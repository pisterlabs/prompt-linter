import re
from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.retrievers.document_compressors.chain_extract import NoOutputParser
from langchain.chains.flare.base import FinishedOutputParser
from typing import Union, Tuple

# Custom output parser
class Nl2ModelParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Response:" in llm_output:
            return AgentFinish(return_values={"output": llm_output.split("Begin!")[-1].strip()},
                log=llm_output,
                )
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            return AgentFinish(return_values={"output": llm_output},
                log=llm_output,
                )
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    

class ModelicaOutputParser(FinishedOutputParser):

    def parse(self, text: str) -> Tuple[str, bool]:
        cleaned = text.strip()
        finished = self.finished_value in cleaned
        response = cleaned.split(self.finished_value)[0]
        return response, finished

class NoModelicaParser(NoOutputParser):
    """Parse outputs that could return a null string of some sort."""

    no_output_str: str = "NO_OUTPUT"

    def parse(self, text: str) -> str:
        cleaned_text = text.strip()
        if cleaned_text == self.no_output_str:
            return ""
        return cleaned_text