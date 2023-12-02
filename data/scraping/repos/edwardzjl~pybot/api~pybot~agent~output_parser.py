import ast
from typing import Generator, Union

from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish


def find_dicts(s: str) -> Generator[tuple[dict, int], None, None]:
    """find dicts in a string
    Modified from <https://stackoverflow.com/a/51862122/6564721>

    Args:
        s (str): source string

    Yields:
        Generator[tuple[dict, int], None, None]: generates a tuple of dict and the index of the last char of the dict
    """
    stack = []  # a stack to keep track of the brackets
    buffer = ""  # a buffer to store current tracking string
    for i, ch in enumerate(s):
        if ch == "{":
            buffer += ch
            stack.append(ch)
        elif ch == "}":
            stack.pop(-1)
            buffer += ch
            if not stack:
                yield ast.literal_eval(buffer), i
                buffer = ""
        elif stack:
            buffer += ch


class JsonOutputParser(AgentOutputParser):
    """Output parser that extracts all dicts in the output and try to parse them into actions.
    Only the first valid action will be returned.
    The AgentOutputParser is a langchain.load.serializable.Serializable which is a pydantic v1 model in the time of writing.
    """

    tool_name_key: str = "tool_name"
    tool_input_key: str = "tool_input"

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        for _dict, i in find_dicts(text):
            if self.tool_name_key in _dict:
                tool_name = _dict.get(self.tool_name_key)
                tool_input = _dict.get(self.tool_input_key, "")
                return AgentAction(tool_name, tool_input, text[: i + 1])
        return AgentFinish({"output": text}, text)

    @property
    def _type(self) -> str:
        return "json"
