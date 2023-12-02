from typing import Any, Dict, List, Union

from langchain.schema import AgentAction, LLMResult
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from utils.console_logger import ConsoleLogger


class CustomStreamCallback(StreamingStdOutCallbackHandler):
    """
    Custom callback handler that uses ConsoleLogger log output & color streamed output from current_stream_color
    """

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """
        Automatically logs "Thinking..." when LLM starts.
        """
        ConsoleLogger.log_thinking()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        ConsoleLogger.log_streaming(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """
        Resets stream color and prints an empty line on LLM stream end.
        """
        print("\n")
        ConsoleLogger.set_default_stream_color()

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """
        Log agents action in magenta
        """
        ConsoleLogger.log_tool(f"Agent action: {action}")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """
        Log tool start and input in magenta
        """
        ConsoleLogger.log_tool(f"Tool started. input_str: {input_str}")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """
        Log tool end and output in magenta
        """
        ConsoleLogger.log_tool(f"Tool ended. output: {output}")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """
        Log tool error in magenta
        """
        ConsoleLogger.log_error(f"Tool error: {error}")
