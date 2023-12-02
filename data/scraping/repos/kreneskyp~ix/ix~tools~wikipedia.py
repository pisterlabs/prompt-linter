from typing import Any
from langchain.tools import WikipediaQueryRun, BaseTool
from langchain.utilities.wikipedia import WikipediaAPIWrapper
from ix.chains.asyncio import SyncToAsyncRun
from ix.chains.loaders.tools import extract_tool_kwargs


class AsyncWikipediaQueryRun(SyncToAsyncRun, WikipediaQueryRun):
    pass


def get_wikipedia(**kwargs: Any) -> BaseTool:
    tool_kwargs = extract_tool_kwargs(kwargs)
    wrapper = WikipediaAPIWrapper(**kwargs)
    return AsyncWikipediaQueryRun(api_wrapper=wrapper, **tool_kwargs)
