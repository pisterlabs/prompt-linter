from enum import Enum
from typing import Callable, List, Optional
import re
from pydantic import BaseModel
from langchain.agents import Tool


class NodeMode(str, Enum):
    ACTION = "action"
    CONDITION = "condition"


class Node(BaseModel):
    name: str
    mode: NodeMode
    description: str
    executor: Callable
    preconditions: Optional[List[str]] = None
    needs_tool: Optional[str] = None
    input_format: Optional[str] = None

    @classmethod
    def from_docstring(cls, fn: Callable):
        docstring = fn.__doc__
        lines = docstring.strip().split("\n")
        fields = {}

        for line in lines:
            # colon is the separator
            match = re.match(r"(.*):(.*)", line.strip())
            if match:
                key, value = match.groups()
                key, value = key.strip(), value.strip()  # remove whitespace
                fields[key] = value.split(
                    ",") if key == "preconditions" else value
        fields["executor"] = fn
        fields["name"] = fn.__name__

        return cls(**fields)


def node_to_tool(node: Node) -> Tool:
    need_tool_str = f"NEEDS TOOL = {node.needs_tool}" if node.needs_tool else ""
    preconditions_str = f"!!! PRECONDITIONS = {', '.join(node.preconditions)}" if node.preconditions else "No preconditions"

    return Tool(
        node.name,
        node.executor,
        f"{node.description}; [{preconditions_str}; {need_tool_str}]; Accepts inputs of the form: {node.input_format}",
    )


def node_tool(fn: Callable) -> Tool:
    node = Node.from_docstring(fn)
    return node_to_tool(node)
