from langchain.tools import BaseTool

from ..state import State


class EchoTool(BaseTool):
    name = "echo"
    description = "An exampel tool that echoes what you input."
    shared_state: State

    def _run(self, query: str) -> str:
        return query
