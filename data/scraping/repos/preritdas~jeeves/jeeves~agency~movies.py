"""
Get info on movies. This tool is inactive because it's currently easier for the 
agent to just Google the movie's name and use Website Answerer on the 
Rotten Tomatoes link. It did this on its own in testing.
"""
from langchain.agents.tools import BaseTool
import rottentomatoes as rt

from typing import Any, Coroutine


class MoviesTool(BaseTool):
    """Get info on movies."""
    name: str = "Movie Info"
    description: str = (
        "Useful for when you want information on movies. "
        "Input should be a string of the movie's name."
    )

    def _run(self, query: str) -> str:
        """Run the tool."""
        movie = rt.Movie(query)
        return str(movie)

    def _arun(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, str]:
        raise NotImplementedError()
