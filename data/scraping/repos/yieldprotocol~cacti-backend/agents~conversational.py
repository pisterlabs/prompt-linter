import re
from typing import Optional, Tuple

from langchain.agents.conversational.base import ConversationalAgent
from langchain.agents.loading import AGENT_TO_CLASS


CUSTOM_AGENT_NAME = "custom-conversational-react-description"


class CustomConversationalAgent(ConversationalAgent):

    def _extract_tool_and_input(self, llm_output: str) -> Optional[Tuple[str, str]]:
        if f"{self.ai_prefix}:" in llm_output:
            return self.ai_prefix, llm_output.split(f"{self.ai_prefix}:")[-1].strip()
        regex = r"Prior Observations: (.*?)[\n]*Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, llm_output)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        prior = match.group(1)
        action = match.group(2)
        action_input = match.group(3)
        action_input = action_input.strip(" ")
        prior = prior.strip(" ")
        action_input = f"{action_input}, given {prior}"
        return action.strip(), action_input



AGENT_TO_CLASS[CUSTOM_AGENT_NAME] = CustomConversationalAgent
