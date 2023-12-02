from langchain.agents import AgentOutputParser
from typing import Union
from langchain.schema import AgentAction, AgentFinish
import re
import json
from fix_busted_json import first_json

class JsonOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        text = llm_output

        print(f"OutputParser input text 1: {text}")

        # repair incorrect JSON formatting
        text = self._fix_common_errors(text)

        found_json = first_json(text)
        print(f"OutputParser foundJson: {found_json}")

        try:
            response = json.loads(found_json)
        except Exception as e:
            print(f"OutputParser failed to parse JSON: {e}")
            print(f"OutputParser failed to parse JSON, original text: {llm_output}")
            print(f"OutputParser failed to parse JSON, text: {text}")
            print(f"OutputParser failed to parse JSON, foundJson: {found_json}")

            if re.search("thought:", text, re.IGNORECASE) or re.search("action:", text, re.IGNORECASE) or re.search("actionInput", text, re.IGNORECASE):
                raise ValueError(f"OutputParser failed to parse JSON, unrecoverable:\n{llm_output}")

            final_answers = {"output": llm_output}
            return AgentFinish(return_values=final_answers, log=text)

        print(f"OutputParser parsed JSON: {response}")

        if re.search("finalresponse", response["action"], re.IGNORECASE):
            final_answers = {"output": response["actionInput"]}
            return AgentFinish(return_values=final_answers, log=text)

        tool = response["action"]

        print(f"OutputParser parsed tool: {tool}")

        tool_input = response["actionInput"]
        if isinstance(tool_input, dict):
            tool_input = json.dumps(tool_input)

        return AgentAction(tool=tool, tool_input=tool_input, log=text)

    def _fix_common_errors(self, text: str) -> str:
        if not "{" in text:
            text = "{" + text
        if not "}" in text:
            text += "}"
        return text
