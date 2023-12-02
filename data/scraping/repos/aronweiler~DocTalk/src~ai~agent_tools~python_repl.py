import logging

from langchain.tools.python.tool import PythonREPLTool as python_repl_tool
from ai.agent_tools.utilities.abstract_tool import AbstractTool

class PythonREPLTool(AbstractTool):
    def configure(
        self, memory=None, override_llm=None, json_args=None
    ) -> None:
        self.tool = python_repl_tool()

    def run(self, query: str) -> str:
        logging.debug("PythonREPLTool got query: " + query)
        return self.tool.run(query)
