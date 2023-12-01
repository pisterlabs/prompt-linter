from langchain.tools import tool


@tool("search", return_direct=True)
def search_api(query: str) -> str:
    """Searches the API for the query."""
    return "Results"
