from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

from ansi import ANSI, Color, Style, dim_multiline
from logger import logger


class EVALCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        pass

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        logger.info(ANSI(f"Entering new chain.").to(Color.green(), Style.italic()))
        logger.info(ANSI("Prompted Text").to(Color.yellow()) + f': {inputs["input"]}\n')

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        logger.info(ANSI(f"Finished chain.").to(Color.green(), Style.italic()))

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        logger.error(
            ANSI(f"Chain Error").to(Color.red()) + ": " + dim_multiline(str(error))
        )

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        pass

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        logger.info(
            ANSI("Observation").to(Color.magenta()) + ": " + dim_multiline(output)
        )
        logger.info(ANSI("Thinking...").to(Color.green(), Style.italic()))

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        logger.error(ANSI("Tool Error").to(Color.red()) + f": {error}")

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Optional[str],
    ) -> None:
        pass

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        logger.info(
            ANSI("Final Answer").to(Color.yellow())
            + ": "
            + dim_multiline(finish.return_values.get("output", ""))
        )
