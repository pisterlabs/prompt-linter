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

scope = "playlist-read-private"
spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))


class GetPlaylistContentsInput(BaseModel):
    """Input for GetPlaylistContentsTool."""

    playlist_id: str = Field(..., description="the id of the playlist to get contents from")


class PlaylistContentsTool(BaseTool):
    """Tool that returns songs contained in a spotify playlist"""

    name: str = "get_spotify_playlist_contents"
    args_schema: Type[BaseModel] = GetPlaylistContentsInput
    description: str = "Returns the songs of a spotify playlist for a given playlist ID."

    def _run(
            self,
            playlist_id: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> list:

        try:
            playlist_contents = spotify.playlist_items(playlist_id)
            songs = []
            if len(playlist_contents['items']) == 0:
                return "Playlist is empty."

            for item in playlist_contents['items']:
                if item["track"] is not None:
                    songs.append(item["track"]["name"] + " by " + item["track"]["artists"][0]["name"] + "with URI: " + item["track"]["uri"])
                else:
                    continue
            if len(songs) == 0:
                return "Playlist is empty."
            else:
                return songs
        except Exception as e:
            print(str(e))
            return {"error": "Unable to get playlist contents."}

    async def _arun(
            self,
            file_path: str,
            text: str,
            append: bool = False,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> dict:
        raise NotImplementedError

