from langchain.tools import DuckDuckGoSearchRun
from ai.agent_tools.utilities.abstract_tool import AbstractTool


class DuckDuckGoTool(AbstractTool):
    def configure(
        self, memory=None, override_llm=None, json_args=None
    ) -> None:
        self.search = DuckDuckGoSearchRun()

    def run(self, query: str) -> str:
        return self.search.run(query)
