from typing import Any, Dict, List, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult


class StreamingQueueCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming to Queue."""

    def __init__(self, q, end, error):
        self.q = q
        self.end = end
        self.error = error

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Run on new LLM token. Streams to Queue."""
        for char in token:
            self.q.put((char,))

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Finish the Queue when the LLM is done."""
        self.q.put((self.end,))

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        print("LLM ERROR", error)
        self.q.put((self.error, error))

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""
        self.q.put((self.error, error))

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
