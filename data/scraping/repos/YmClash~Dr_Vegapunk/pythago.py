from langchain.agents import Tool
from spotify import start_playing_song_by_name


def tool_start_playing_song_by_name():
    return Tool(name="Play a song given the name", func=lambda song_name:start_playing_song_by_name(song_name),
                description=f"""given a song name ,start playing a song, ActionInput is a string of song_name""",
                return_direct=True)
