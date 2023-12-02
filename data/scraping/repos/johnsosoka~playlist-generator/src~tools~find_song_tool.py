from typing import Type
import spotipy
from langchain.tools import BaseTool
from typing import Optional, Type

from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from spotipy.oauth2 import SpotifyOAuth

scope = "user-library-read playlist-modify-public playlist-modify-private"
spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

class FindSongInput(BaseModel):
    """Input for WriteFileTool."""

    artist: str = Field(..., description="name of artist/band")
    title: str = Field(..., description="The title of the song")

class FindSongTool(BaseTool):
    """Tool that finds a song on Spotify."""

    name: str = "find_song"
    args_schema: Type[BaseModel] = FindSongInput
    description: str = "Finds a song on Spotify. Returns the Spotify URI if found."

    def _run(
            self,
            artist: str,
            title: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        results = spotify.search(q=f"track:{title} artist:{artist}", type="track")

        if results["tracks"]["items"]:
            top_result_item = results["tracks"]["items"][0]
            return top_result_item["uri"]
        else:
            # Handle the case where the search didn't return any results
            message = f"No results found for track:{title} artist:{artist}"
            return message

    async def _arun(
            self,
            artist: str,
            title: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError