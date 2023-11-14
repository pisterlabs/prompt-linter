"""Chain for selecting a MRKL tool for use."""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate
from langchain.tools.base import BaseTool

from langchain_contrib.utils import call_llm

from .prompt import MrklPromptSelector

FINAL_ANSWER_ACTION = "Final Answer:"


class MrklPickActionChain(Chain):
    """MRKL chain that selects a tool for use."""

    llm: BaseLanguageModel
    """LLM to run this iteration of the MRKL agent with."""
    prompt: BasePromptTemplate
    """Prompt for a single response of the LLM."""
    observation_prefix: str = "Observation: "
    """Observation prefix to stop the MRKL at."""
    choice_key: str = "choice"
    """Output key for which action the LLM chose."""
    action_input_key: str = "action_input"
    """Output key for the action's input."""

    @classmethod
    def from_tools(
        cls,
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        prompt: Optional[BasePromptTemplate] = None,
        embed_scratchpad: bool = True,
    ) -> MrklPickActionChain:
        """Instantiate the MRKL action chain from a list of tools."""
        if prompt is None:
            prompt = (
                MrklPromptSelector()
                .get_custom_prompt(llm, embed_scratchpad=embed_scratchpad)
                .permissive_partial(tools=tools)
            )
        return cls(llm=llm, prompt=prompt)

    @property
    def input_keys(self) -> List[str]:
        """Input keys this chain expects."""
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Output keys this chain expects."""
        return ["observation", self.choice_key, self.action_input_key]

    def get_action_and_input(self, llm_output: str) -> Tuple[str, str, str]:
        """Parse out the action and input from the LLM output.

        Copied and edited from langchain/agents/mrkl/base.py
        """
        if FINAL_ANSWER_ACTION in llm_output:
            return "", "Final Answer", llm_output.split(FINAL_ANSWER_ACTION)[-1].strip()
        regex = r"(.*)[\n]*Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        observation = match.group(1).strip()
        action = match.group(2).strip()
        action_input = match.group(3)
        return observation, action, action_input.strip(" ").strip('"')

    def get_chat_action_and_input(self, llm_output: str) -> Tuple[str, str, str]:
        """Parse out the action and input from the LLM output.

        Copied and edited from langchain/agents/chat/base.py
        """
        if FINAL_ANSWER_ACTION in llm_output:
            return "", "Final Answer", llm_output.split(FINAL_ANSWER_ACTION)[-1].strip()
        try:
            match = re.search(r"Thought:(.*)\n*Action:", llm_output, re.DOTALL)
            assert match is not None
            observation = match.group(1).strip()
            _, action, _ = llm_output.split("```")
            response = json.loads(action.strip())
            return observation, response["action"], response["action_input"]

        except Exception:
            raise ValueError(f"Could not parse LLM output: {llm_output}")

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Get the LLM to pick an action and its input."""
        llm_response = call_llm(
            self.llm,
            self.prompt.format_prompt(**inputs),
            stop=[f"\n{self.observation_prefix.rstrip()}"],
        )
        if isinstance(self.llm, BaseLLM):
            observation, action, input = self.get_action_and_input(llm_response)
        else:
            observation, action, input = self.get_chat_action_and_input(llm_response)
        return {
            "observation": observation,
            self.choice_key: action,
            self.action_input_key: input,
        }
