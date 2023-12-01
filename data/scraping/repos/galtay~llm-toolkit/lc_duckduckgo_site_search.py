"""
DuckDuckGo search restricted to one site
https://github.com/deedy5/duckduckgo_search


before using a specific site, check if you get results in a web browser
https://duckduckgo.com/
https://help.duckduckgo.com/settings/params/

"""
from langchain.agents import Tool
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import DuckDuckGoSearchResults
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from duckduckgo_search import DDGS
from rich import print



def get_ddgs_results(query: str, site: str, region="wt-wt", safesearch="moderate", timelimit="y"):
    with DDGS() as ddgs:
        results = list(ddgs.text(
            f"site:{site} {query}",
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
        ))
    return results

site = "en.wikipedia.org"
query = "physics"
ddgs_results = get_ddgs_results(query, site)


class DuckDuckGoSearchSiteAPIWrapper(DuckDuckGoSearchAPIWrapper):

    site: str

    def get_snippets(self, query: str) -> list[str]:
        from duckduckgo_search import DDGS
        no_results_msg = "No good DuckDuckGo Search Result was found"

        with DDGS() as ddgs:
            results = ddgs.text(
                f"site:{self.site} {query}",
                region=self.region,
                safesearch=self.safesearch,
                timelimit=self.time,
            )
            if results is None:
                return [no_results_msg]
            snippets = []
            for i, res in enumerate(results, 1):
                if res is not None:
                    snippets.append(res["body"])
                if len(snippets) == self.max_results:
                    break
        return snippets




    def run(self, query: str) -> str:
        snippets = self.get_snippets(query)
        return " ".join(snippets)


class DuckDuckGoSearchSiteRun(DuckDuckGoSearchRun):
    pass
    


api_wrapper = DuckDuckGoSearchSiteAPIWrapper(site=site)
search_res = DuckDuckGoSearchResults(api_wrapper=api_wrapper)
search_run = DuckDuckGoSearchRun(
    return_direct = False,
    verbose = False,
    api_wrapper = api_wrapper,
)


def search_site_wrapper(input_text: str, site: str) -> str:
    """Specify a site like 'webmd.com' or 'congress.gov' or 'wikipedia.org'"""
    search_results = search_run.run(f"site:{site} {input_text}")
    return search_results


tools = [
    Tool(
        name="Search",
        func=search_run.run,
        description="useful for when you need to answer questions about current events",
    )
]
