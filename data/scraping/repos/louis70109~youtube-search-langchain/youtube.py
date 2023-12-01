from ast import literal_eval
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import YouTubeSearchTool
import logging


class YoutubeDefineInput(BaseModel):
    """Youtube video recommendation."""

    title: str = Field(
        ...,
        description="video title which will be recommend to user.")


class FindYoutubeVideoTool(BaseTool):
    name = "youtube_search"
    description = "Find recommendation video from Youtube"

    def _run(self, title: str):
        logging.info("Youtube")
        logging.info('標題：'+title)
        tool = YouTubeSearchTool()
        youtube_str = tool.run(title)  # force change str to list
        youtube_list = literal_eval(youtube_str)
        for i in range(len(youtube_list)):
            youtube_list[i] = f'https://www.youtube.com/{youtube_list[i]}'
        return youtube_list

    args_schema: Optional[Type[BaseModel]] = YoutubeDefineInput
