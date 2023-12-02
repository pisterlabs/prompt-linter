from __future__ import annotations

import re
from typing import Union

from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish

from agentverse.parser import OutputParseError, output_parser_registry


@output_parser_registry.register("math_problem_2players_tools")
class MathProblem2PlayersToolsParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:

        cleaned_output = text.strip()
        cleaned_output = re.sub(r'\n+', '\n', cleaned_output)
        cleaned_output = cleaned_output.split('\n')
        if not (len(cleaned_output) == 2 and
                # cleaned_output[0].startswith("THOUGHT:") and
                cleaned_output[0].startswith("ACTION:") and
                cleaned_output[1].startswith("ACTION INPUT:")):
            print(text)
            raise OutputParseError("Output Format Error")
        action = cleaned_output[0][len("ACTION:"):].strip()
        action_input = cleaned_output[1][len("ACTION INPUT:"):].strip()
        if action in ["Speak"]:
            return AgentFinish({"output": action_input}, text)
        else:
            return AgentAction(action, action_input, text)
