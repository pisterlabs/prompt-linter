from schema import Thread
from langchain.tools import Tool
from langchain import PromptTemplate

class Skill(Thread):
    
    tool: Tool
    prompt: PromptTemplate




    def __init__(self) -> None:
        pass

    def _call(self) -> dict[str, Any]:
        return self.chain._call()