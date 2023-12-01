from typing import Optional, Union, Dict

from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from ai.agent_tools.utilities.abstract_tool import AbstractTool


class WikipediaTool(AbstractTool):
    def configure(
        self,
        memory=None,
        override_llm=None,
        json_args: Optional[Union[Dict, None]] = None,
    ) -> None:
        top_k_results = 1
        doc_content_chars_max = 4000

        if json_args:
            top_k_results = json_args.get("top_k_results", 1)
            top_k_results = json_args.get("doc_content_chars_max", 4000)

        self.wikipedia = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(
                top_k_results=top_k_results, doc_content_chars_max=doc_content_chars_max
            )
        )

    def run(self, query: str) -> str:
        try:
            result = self.wikipedia.run(query)

            return result
        except:
            return "Could not reach wikipedia"
