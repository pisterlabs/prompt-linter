from langchain.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain.utilities import DuckDuckGoSearchAPIWrapper

wrapper = DuckDuckGoSearchAPIWrapper(region="jp-jp")

search = DuckDuckGoSearchResults(api_wrapper=wrapper, backend="api")
ret = search.run("Whos Luke in street figher 6")
print(ret)

