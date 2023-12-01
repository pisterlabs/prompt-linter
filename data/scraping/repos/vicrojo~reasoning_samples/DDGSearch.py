from langchain.utilities  import DuckDuckGoSearchAPIWrapper
from langchain.tools import DuckDuckGoSearchResults


def get_search_results(query: str):
    """Searches for query on google."""
    wrapper = DuckDuckGoSearchAPIWrapper()
    search = DuckDuckGoSearchResults(api_wrapper=wrapper)
    res = search.run(f"{query}")
    return res