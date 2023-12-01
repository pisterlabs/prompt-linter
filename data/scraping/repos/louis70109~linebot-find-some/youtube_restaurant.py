from ast import literal_eval
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import YouTubeSearchTool


class YoutubeDefineInput(BaseModel):
    """Youtube Restaurant recommendation."""

    title: str = Field(
        ...,
        description="Restaurant title which will be recommend to user.")


class FindYoutubeVideoTool(BaseTool):
    name = "find_restaurant_youtube"
    description = "Find recommendation restaurant from Youtube"

    def _run(self, title: str):
        print("Youtube")
        print('標題：'+title)
        tool = YouTubeSearchTool()
        youtube_str = tool.run(title)  # force change str to list
        youtube_list = literal_eval(youtube_str)
        for i in range(len(youtube_list)):
            youtube_list[i] = youtube_list[i]
        return youtube_list

    args_schema: Optional[Type[BaseModel]] = YoutubeDefineInput