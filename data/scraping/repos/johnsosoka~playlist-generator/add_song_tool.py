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


class AddSongInput(BaseModel):
    """Input for WriteFileTool."""

    uri: str = Field(..., description="the uri of the song to add")
    playlist_id: str = Field(..., description="the id of the playlist to add the song to")


class AddSongTool(BaseTool):
    """Tool that adds a song to a spotify playlist. for a given URI."""

    name: str = "add_song"
    args_schema: Type[BaseModel] = AddSongInput
    description: str = "Adds a song to a spotify playlist for a given URI."

    def _run(
            self,
            uri: str,
            playlist_id: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:

        try:
            spotify.playlist_add_items(playlist_id, [uri])
            return f"Song added successfully to playlist."
        except Exception as e:
            print(str(e))
            return "Unable to add song to playlist. Error: "

    async def _arun(
            self,
            file_path: str,
            text: str,
            append: bool = False,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError
