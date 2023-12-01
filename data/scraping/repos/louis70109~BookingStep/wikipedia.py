from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import datetime
import urllib
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

class WikiInput(BaseModel):
    """Input for Google Calendar Generator."""

    title: str = Field(
        ...,
        description="Wikipedia Title symbol ")
    link: str = Field(
        ...,
        description="Wikipedia url")


class WikiTool(BaseTool):
    name = "find_wikipedia_information"
    description = f"Use wikipedia resources to find unknown information."

    def _run(self, title: str, link: str):
        print("Wiki")
        print('標題：'+title)
        print('描述：'+link)
        
        

        return title, link

    args_schema: Optional[Type[BaseModel]] = WikiInput
