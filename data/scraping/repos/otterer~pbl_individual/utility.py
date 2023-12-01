
from typing import Any, Dict, List, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction


class AcademicPeriod:
    def __init__(self, quarter: str, start_week: int, end_week: int):
        self.quarter = quarter
        self.weeks = [f"{i:02d}" for i in range(start_week, end_week + 1)]


ACADEMIC_PERIODS = {
    "4": AcademicPeriod("1Q", 1, 4),
    "5": AcademicPeriod("1Q", 5, 9),
    "6": AcademicPeriod("2Q", 1, 4),
    "7": AcademicPeriod("2Q", 5, 9),
    "8": AcademicPeriod("2Q", 9, 10),
    "9": AcademicPeriod("3Q", 1, 4),
    "10": AcademicPeriod("3Q", 5, 9),
    "11": AcademicPeriod("4Q", 1, 4),
    "12": AcademicPeriod("4Q", 5, 9),
    "1": AcademicPeriod("4Q", 9, 10),
}


class SlackCallbackHandler(BaseCallbackHandler):
    def __init__(self, say_function):
        self.say = say_function
        self.token_count = 0
        self.content = []

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        if self.token_count < 100:
            self.token_count += 1
            self.content.append(token)
        else:
            self.token_count = 0
            print(''.join(self.content))
            self.say(''.join(self.content))
            self.content = []

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        print(f"on_chain_start {serialized['name']}")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        print(f"on_tool_start {serialized['name']}")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        print(f"on_agent_action {action}")
