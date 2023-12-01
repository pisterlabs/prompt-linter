from typing import Tuple, Any, Dict, List, Optional, Union, Callable
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AgentAction, AgentFinish, LLMResult
from langchain.callbacks.base import BaseCallbackHandler, CallbackManager
import logging

from llm_oracle.markets.custom import CustomEvent
from llm_oracle.agents.agent_tools import ToolAgentv3
from llm_oracle.agents.agent_basic import BasicAgentv3


SYSTEM_VALIDATE_PROMPT = """
You are the validation system to an application that makes binary predictions about world events.

Given the users prediction question, validate if it meets the criteria:
* Must have clear end time ("Will AI take over the world?" is invalid)
* Must be a prediction question ("Who are you?" is invalid)
* Must be binary ("When will xyz occur?", "What color will xyz be?" are invalid)
* It does not need to be a world event

Respond only "VALID" if valid, or a brief explanation as to why the prediction question was invalid and provide a similar question they could ask that would be valid.

Your response will be shown as an alert on the UI to the user if not valid.
"""


def validate_question(question: str) -> Tuple[bool, str]:
    llm = ChatOpenAI(
        model_name="gpt-4",
        request_timeout=120,
        max_retries=10,
        temperature=0.0,
    )
    result = llm([SystemMessage(content=SYSTEM_VALIDATE_PROMPT), HumanMessage(content=question)]).content
    logging.debug(f"Question validation result {result}")
    if "VALID" in result:
        return (False, None)
    else:
        return (True, result)


def run_gpt4_agent(temperature: float, question: str, log_callback: Callable[[str], None]) -> int:
    class LoggingCallback(LLMEventLoggingCallback):
        def write_log(self, text: str):
            log_callback(text)

    callback_manager = CallbackManager([LoggingCallback()])

    llm = ChatOpenAI(
        model_name="gpt-4",
        request_timeout=120,
        max_retries=10,
        temperature=temperature,
        callback_manager=callback_manager,
    )
    tool_llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        request_timeout=120,
        max_retries=10,
        temperature=temperature,
    )
    event = CustomEvent(
        question,
        None,
    )
    agent = ToolAgentv3(model=llm, tool_model=tool_llm, callback_manager=callback_manager, use_proxy=False)
    return int(agent.predict_event_probability(event) * 100)


def run_gpt3_agent(temperature: float, question: str, log_callback: Callable[[str], None]) -> int:
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        request_timeout=120,
        max_retries=10,
        temperature=temperature * 0.3,
    )
    event = CustomEvent(
        question,
        None,
    )
    agent = BasicAgentv3(model=llm, output_callback=log_callback)
    return int(agent.predict_event_probability(event) * 100)


class LLMEventLoggingCallback(BaseCallbackHandler):
    def write_log(self, text: str):
        logging.debug(text)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        pass

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        pass

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        pass

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        pass

    def on_agent_action(self, action: AgentAction, color: Optional[str] = None, **kwargs: Any) -> Any:
        self.write_log(action.log.split("Action:")[0].strip())

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        pass

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        pass

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Optional[str],
    ) -> None:
        pass

    def on_agent_finish(self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any) -> None:
        self.write_log(finish.log)


MODEL_RUN_FUNCTIONS = {
    "gpt3": run_gpt3_agent,
    "gpt4": run_gpt4_agent,
}

MODEL_COSTS = {"gpt3": 1, "gpt4": 10}

MODELS_DEMO_SUPPORTED = ["gpt3"]
