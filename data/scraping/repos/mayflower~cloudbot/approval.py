from typing import Any, Callable, Dict, Optional
from uuid import UUID
from termcolor import colored
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import ChatOpenAI


from langchain.callbacks.base import BaseCallbackHandler


def _default_true(_: Dict[str, Any]) -> bool:
    return True


class HumanRejectedException(Exception):
    """Exception to raise when a person manually review and rejects a value."""


class ApprovalCallBackHandler(BaseCallbackHandler):
    """Callback for manually validating values."""

    raise_error: bool = True
    explainer: BaseChatModel

    def __init__(
        self,
        approve: Callable[[Any], bool] = _default_true,
        should_check: Callable[[Dict[str, Any]], bool] = _default_true,
    ):
        self._approve = self.approve
        self._should_check = should_check
        self.explainer = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if self._should_check(serialized) and not self._approve(input_str):
            raise HumanRejectedException(
                f"Inputs {input_str} to tool {serialized} were rejected."
            )

    def approve(self, _input: str) -> bool:
        explanation = self.explainer.predict(
            "Please explain with simple words what the following command does, what the expected outcome is and if there are any risks  involved:\n"
            + _input,
        )
        msg = (
            colored("To answer i would like to run this command: ", "light_green")
            + colored(_input, "white")
            + "\n"
        )
        msg += colored(explanation + "\n", "light_grey")
        msg += colored(
            "If this is ok please approve it with 'y' or 'yes':",
            "light_red",
        )
        resp = input(msg)
        result = resp.lower() in ("yes", "y")
        if result:
            print(colored("Executing " + _input, "light_grey"))
        return resp.lower() in ("yes", "y")
