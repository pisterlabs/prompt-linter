import logging

from langchain.agents import Tool
from langchain.tools import BaseTool, StructuredTool
import importlib
from typing import cast
from ai.agent_tools.utilities.tool_header import ToolHeader
from ai.agent_tools.utilities.abstract_tool import AbstractTool


def create_tool(
    tool_json=None, memory=None, override_llm=None
):
    if tool_json is None:
        raise Exception("Tool JSON must be provided")


    header = ToolHeader(tool_json)

    try:
        module = importlib.import_module(header.tool_module_name)

        # dynamically instantiate the tool based on the parameters
        tool_instance = getattr(module, header.tool_class_name)()
        typed_instance = cast(AbstractTool, tool_instance)

        if "arguments" in tool_json:
            tool_instance.configure(
                memory=memory,
                override_llm=override_llm,
                json_args=tool_json["arguments"],
            )
        else:
            tool_instance.configure(
                memory=memory, override_llm=override_llm
            )

        tool = Tool(
            name=header.tool_name,
            func=typed_instance.run,
            description=header.tool_description,
        )

        return tool
    except Exception as e:
        logging.debug("Error creating tool: " + str(e))

    return None
