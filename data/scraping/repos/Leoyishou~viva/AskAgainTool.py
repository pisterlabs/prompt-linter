from typing import Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

def askAgain(errorMessage):
    print("请补全参数")

class AskAgainToolInput(BaseModel):
    errorMessage : str = Field()


class AskAgainTool(BaseTool):
    name = "ask again"
    description = "ask again, complete the args which are absent"
    args_schema : Type[BaseModel] = AskAgainToolInput

    def _run(self, errorMessage : str):
        askAgain(errorMessage)
    def _arun(self, url: str):
        raise NotImplementedError("error here")
