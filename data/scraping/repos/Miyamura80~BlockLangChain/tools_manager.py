from typing import List, Tuple
import numpy as np
from memory import Memory
from langchain.tools import BaseTool
from langchain.chat_models.base import BaseChatModel
from embedding_utils import get_embedding, get_embedding_scores


def select_tools(
    mem: Memory,
    query: str,
    tools_with_embeddings: List[Tuple[BaseTool, np.ndarray]],
    max_num_suggestions=3,
    chat: BaseChatModel = None
) -> List[BaseTool]:
    memory_for_query = mem.generate_context(query)
    query = (
        f"We want to find the best tools that handle {query}. The available tools are:\n"
        f""
        f"So far we have observed the following:\n"
        f"{memory_for_query}"
    )

    if chat is not None:
        query = chat.call_as_llm(query)

    ordered_tools = sorted(
        get_embedding_scores(
            tools_with_embeddings,
            query,
            decay=1.0,
            lb_decay=1.0
        )
    )[::-1]

    return [tool for score, tool in ordered_tools[:max_num_suggestions]]


def get_tool_embedding(tool: BaseTool) -> np.ndarray:
    return get_embedding(
        f"This is a prompt for the {tool.name} tool. The description of the tool is given below: {tool.description}\n"
    )


def prepare_tools(tools: List[BaseTool]) -> List[Tuple[BaseTool, np.ndarray]]:
    return [
        (tool, get_tool_embedding(tool))
        for tool in tools
    ]