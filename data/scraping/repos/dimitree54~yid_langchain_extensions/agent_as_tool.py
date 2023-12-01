from typing import Any, Dict, Callable, Tuple

from langchain.agents import AgentExecutor
from langchain.tools import BaseTool


class AgentAsTool(BaseTool):
    executor: AgentExecutor
    adapter: Callable[[Tuple[Any], Dict[str, Any]], str]

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        args, kwargs = self.adapter(args, kwargs)
        return self.executor.run(*args, **kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        args, kwargs = self.adapter(args, kwargs)
        return await self.executor.arun(*args, **kwargs)
