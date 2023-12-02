from tools.base_tool import BaseTool
from utils import prompts
from langchain.tools import DuckDuckGoSearchRun

class NetworkSearch(DuckDuckGoSearchRun):

    def __init__(self, **kwargs) -> None:
        super().__init__()

    @prompts(name="Network Search", 
             desc="useful when you want to search for a network architecture."
             "The input is a text, which is the query that user want to search.")
    def inference(self, text):
        return self._run(text)