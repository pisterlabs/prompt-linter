from langchain.agents import AgentOutputParser
from typing import Union
from langchain.schema import AgentAction, AgentFinish
import re

class ThreeUnderscoreOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        text = llm_output

        print(f"OutputParser input text 1: {text}")

        # repair incorrect end_action_input, end_action, end_thought, end_response
        text = self._repair_tags(text)

        # sometimes the llm forgets to give any of the tags, especially when giving the final response
        # or sometimes it forgets the closing tags, so we try to repair it here
        text = self._add_missing_tags(text)

        print(f"OutputParser input text 2: {text}")

        # Parse out the action and action input
        regex = r"___start_action___(.*)___end_action___.*___start_action_input___(.*)___end_action_input___"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)

        # Check if agent should finish
        if re.search("finalresponse", action, re.IGNORECASE):
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": action_input},
                log=llm_output,
            )

        print(f"OutputParser parsed tool: {action}")

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)

    def _repair_tags(self, text: str) -> str:
        replacements = [
            (f"___action_input___", f"___end_action_input___"),
            (f"___action___", f"___end_action___"),
            (f"___thought___", f"___end_thought___"),
            (f"___response___", f"___end_response___"),
            (f"___action_input___", f"___start_action_input___"),
            (f"___action___", f"___start_action___"),
            (f"___thought___", f"___start_thought___"),
            (f"___response___", f"___start_response___"),
        ]

        for old, new in replacements:
            if old in text and new in text:
                text = text.replace(old, new)

        return text

    def _add_missing_tags(self, text: str) -> str:
        if "___start_action_input___" not in text:
            text = "___start_action_input___" + text
        if "___start_action___" not in text:
            text = "___start_action___finalresponse___end_action___" + text
        if "___start_thought___" not in text:
            text = "___start_thought___No thought.___end_thought___" + text
        if "___start_response___" not in text:
            text = "___start_response___" + text

        if "___end_action_input___" not in text:
            text += "___end_action_input___"
        if "___end_response___" not in text:
            text += "___end_response___"

        return text
