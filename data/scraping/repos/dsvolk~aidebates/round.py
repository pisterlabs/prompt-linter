from typing import Dict

from langchain.memory import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate

from src.gradio import stream


def _make_prompt_template(prompt: str, history: ChatMessageHistory) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", prompt)] + history.messages)  # type: ignore


class DebateRound:
    def __init__(self, motion: str, gov_prompt: str, opp_prompt: str, llm_params: Dict):
        self.motion = motion
        self.gov_prompt = gov_prompt
        self.opp_prompt = opp_prompt

        self.gov_history = ChatMessageHistory()
        self.opp_history = ChatMessageHistory()

        self.llm_params = llm_params

    def gov_speech(self):
        prompt_template = _make_prompt_template(self.gov_prompt, self.gov_history)

        content = ""
        for content in stream(prompt_template, self.llm_params, {"motion": self.motion}):
            yield content

        self.gov_history.add_ai_message(content)
        self.opp_history.add_user_message(content)

    def opp_speech(self):
        prompt_template = _make_prompt_template(self.opp_prompt, self.opp_history)

        content = ""
        for content in stream(prompt_template, self.llm_params, {"motion": self.motion}):
            yield content

        self.gov_history.add_user_message(content)
        self.opp_history.add_ai_message(content)
