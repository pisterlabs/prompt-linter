import ast
from typing import Any, Callable, Dict, Optional
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler


# TODO: add approval for APIs, write_file
def _should_check(serialized_obj: Dict[str, Any]) -> bool:
    # Only require approval on ShellTool.
    return serialized_obj.get("name") in ("terminal", "python_repl")


def _approve(serialized_obj: Dict[str, Any], _input: str) -> bool:
    if _input == serialized_obj.get("name") == "python_repl":
        ast.literal_eval(_input)
    msg = "Do you approve of the following input? " "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no."
    msg += "\n" + str(serialized_obj) + "\n" + _input + "\n"
    resp = input(msg)
    return resp.lower() in ("yes", "y")


class HumanRejectedException(Exception):
    """Exception to raise when a person manually review and rejects a value."""


class HumanApprovalCallbackHandler(BaseCallbackHandler):
    """Callback for manually validating values."""

    raise_error: bool = True

    def __init__(
        self,
        approve: Callable[[Dict[str, Any], Any], bool] = _approve,
        should_check: Callable[[Dict[str, Any]], bool] = _should_check,
    ):
        self._approve = approve
        self._should_check = should_check

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if self._should_check(serialized) and not self._approve(serialized, input_str):
            raise HumanRejectedException(f"Inputs {input_str} to tool {serialized} were rejected.")
