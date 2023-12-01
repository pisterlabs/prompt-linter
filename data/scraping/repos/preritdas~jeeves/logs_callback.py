"""Create a logging callback handler for the agent."""
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks import StdOutCallbackHandler

import logging
from logging import Logger
from logging.handlers import SysLogHandler
import re

from typing import Dict, Any, List, Optional, Union
from langchain.schema import LLMResult, AgentAction, AgentFinish

from keys import KEYS
from config import CONFIG


def extract_log_items(log: str, fields: list[str]) -> list[str]:
    """
    Takes a log and extracts the fields specified in the fields list.
    Removes spaces from all field names.

    Args:
        log (str): The log to extract from.
        fields (list[str]): The fields to extract. Don't include the colon.

    Returns:
        list[str]: The extracted fields as full strings.

    Example: if the log is "This: something That: something else" then
    extract_log_items(log, ["This", "That"]) will return
    ["This: something", "That: something else"]

    Spaces are removed from the field names, so "Action Input" becomes "ActionInput"
    in the logs. This is to make the logs more readable. So when sorting and checking
    for the fields, we remove the spaces from the field names (check_fields).
    """
    # Regular expression to match "Thought:", "Action:", and "Action Input:"
    fields = [f + ":" for f in fields]
    pattern = f"({'|'.join(fields)})"

    # Split the string using the pattern and filter out empty strings
    split_string = [s.strip() for s in re.split(pattern, log) if s.strip()]

    # Combine the matched expressions with their corresponding text, including a space after the colon
    logs: list[str] = [
        split_string[i].replace(" ", "") + " " + split_string[i + 1]
        for i in range(0, len(split_string), 2)
    ]

    # Sort the logs in the order of the fields
    check_fields: list[str] = [f.replace(" ", "") for f in fields]
    return sorted(logs, key=lambda x: check_fields.index(x.split(":")[0] + ":"))


class AgentLoggingCallbackHandler(BaseCallbackHandler):
    """
    Callback Handler that logs instead of printing.
    Specific for agents, as it uses agent terminology in the logs.
    """
    def __init__(self, logger: Logger, uid: str) -> None:
        """Initialize callback handler."""
        self.logger = logger
        self.uid = uid

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Do nothing."""
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Do nothing."""
        pass

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized["id"][-1]
        self.logger.info(f"{self.uid}: AgentStart: Entering new {class_name} chain...")
        self.logger.info(f"{self.uid}: UserInput: {inputs['input']}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        self.logger.info(f"{self.uid}: AgentFinish: Finished chain.")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Do nothing."""

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action."""
        log_items = extract_log_items(action.log, ["Thought", "Action", "Action Input"])

        # Log the result
        for result in log_items:
            self.logger.info(f"{self.uid}: {result}")

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        self.logger.info(f"{self.uid}: {observation_prefix}{output}")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when agent ends."""
        self.logger.info(f"{self.uid}: {text}")

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        # If no tools were used
        if "Final Answer" in finish.log and "Thought" not in finish.log:
            self.logger.info(f"{self.uid}: {finish.log.splitlines()[1]}")
            return

        log_items = extract_log_items(finish.log, ["Thought", "Final Answer"])

        # Log the result
        for result in log_items:
            self.logger.info(f"{self.uid}: {result}")


# ---- Logging ----

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)
handler = SysLogHandler(address=(KEYS.Papertrail.host, KEYS.Papertrail.port))
logger.addHandler(handler)


def create_callback_handlers(uid: str) -> list[BaseCallbackHandler]:
    """
    Create a Callback Manager with all the handlers based on the uid. The uid is used
    to separate entries in the logs, so a unique CallbackManager should be used for each agent run.
    """
    # Log to console and to Papertrail
    logging_callback = AgentLoggingCallbackHandler(logger=logger, uid=str(uid))
    callback_handlers = [logging_callback]

    # Log to console as well if configured
    if CONFIG.GPT.console_agent:
        callback_handlers.append(StdOutCallbackHandler())

    return callback_handlers
