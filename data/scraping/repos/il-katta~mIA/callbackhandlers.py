import json
import logging
from multiprocessing import Queue
from typing import Dict, Any, List, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult, AgentAction

__all__ = ["OnStream", "StreamMessage"]


class StreamMessage:
    def __init__(self, type: str, data: Any):
        self.type = type
        self.data = data


class OnStream(BaseCallbackHandler):
    def __init__(self, queue: Queue):
        self._queue = queue
        self._logger = logging.getLogger("OnStream")

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        self._logger.debug(f"LLM started with prompts: {prompts} ( serialized: {serialized} )")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self._logger.debug(f"LLM new token: '{token}'")
        self._queue.put(StreamMessage("token", token))

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self._logger.debug(f"LLM ended with response: {response}")
        self._queue.put(StreamMessage("llm_end", None))

    def on_llm_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        self._logger.error(f"LLM error: {error}")
        self._queue.put(StreamMessage("llm_error", error))

    def on_chain_start(
            self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        self._logger.debug(f"Chain started with inputs: {inputs} ( serialized: {serialized} )")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        self._logger.debug(f"Chain ended with outputs: {outputs}")
        # self._queue.close()
        if "response" in outputs:
            self._queue.put(StreamMessage("response", outputs["response"]))
        self._queue.put(StreamMessage("chain_end", None))

    def on_chain_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""
        self._logger.error(f"Chain error: {error}")
        self._queue.put(StreamMessage("chain_error", error))

    def on_tool_start(
            self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        self._logger.debug(f"Tool started with input: {input_str} ( serialized: {serialized} )")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        self._logger.debug(f"Agent action: {action}")
