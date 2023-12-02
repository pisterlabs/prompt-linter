from typing import List

from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper


class GoogleSearch:
    def __init__(self):
        search = GoogleSearchAPIWrapper()

        self.tool = Tool.from_function(
            name="Google_Search",
            description="Search Google for recent results.",
            func=search.run,
        )

    def get_tools(self) -> List[Tool]:
        return [self.tool]
