from typing import Any, Dict, Optional, Sequence, Type, TypeVar

import json
from json import JSONDecodeError

from halo import Halo
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, FunctionMessage
from langchain.tools import BaseTool
from langchain.tools.render import format_tool_to_openai_function
from pydantic import BaseModel

from chatflock.errors import FunctionNotFoundError
from chatflock.utils import fix_invalid_json


def execute_chat_model_messages(
    chat_model: BaseChatModel,
    messages: Sequence[BaseMessage],
    chat_model_args: Optional[Dict[str, Any]] = None,
    tools: Optional[Sequence[BaseTool]] = None,
    spinner: Optional[Halo] = None,
) -> str:
    chat_model_args = chat_model_args or {}

    if "functions" in chat_model_args:
        raise ValueError(
            "The `functions` argument is reserved for the "
            "`execute_chat_model_messages` function. If you want to add more "
            "functions use the `functions` argument to this method."
        )

    if tools is not None and len(tools) > 0:
        chat_model_args["functions"] = [format_tool_to_openai_function(tool) for tool in tools]

    function_map = {tool.name: tool for tool in tools or []}

    all_messages = list(messages).copy()

    last_message = chat_model.predict_messages(all_messages, **chat_model_args)
    function_call = last_message.additional_kwargs.get("function_call")

    while function_call is not None:
        function_name = function_call["name"]
        if function_name in function_map:
            tool = function_map[function_name]
            args = function_call["arguments"]

            if spinner is not None:
                if hasattr(tool, "progress_text"):
                    progress_text = tool.progress_text
                else:
                    progress_text = f"Executing function `{function_name}`..."

                spinner.start(progress_text)

            try:
                args = json.loads(args)
                result = tool.run(args)
            except JSONDecodeError as e:
                # Try to fix the JSON manually before giving up
                try:
                    args = fix_invalid_json(args)
                    args = json.loads(args)
                    result = tool.run(args)
                except JSONDecodeError as e:
                    result = f"Error decoding args for function: {e}"
            except Exception as e:
                result = f"Error executing function: {e}"

            all_messages.append(
                FunctionMessage(
                    name=function_name,
                    content=f"The function execution returned:\n```{str(result).strip()}```" or "None",
                )
            )

            last_message = chat_model.predict_messages(all_messages, **chat_model_args)
            function_call = last_message.additional_kwargs.get("function_call")
        else:
            raise FunctionNotFoundError(function_name)

    return str(last_message.content)


PydanticType = TypeVar("PydanticType", bound=Type[BaseModel])


def pydantic_to_openai_function(
    pydantic_type: PydanticType, function_name: Optional[str] = None, function_description: Optional[str] = None
) -> Dict[str, Any]:
    base_schema = pydantic_type.model_json_schema()
    del base_schema["title"]
    del base_schema["description"]

    description = function_description if function_description is not None else (pydantic_type.__doc__ or "")

    return {"name": function_name or pydantic_type.__name__, "description": description, "parameters": base_schema}
