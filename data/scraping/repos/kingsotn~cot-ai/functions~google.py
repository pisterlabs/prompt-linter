from typing import List
from .function import Function
from langchain.utilities import GoogleSearchAPIWrapper
from utils.openai_utils import get_turbo_response

class GoogleSearch(Function):

    search: GoogleSearchAPIWrapper
    
    def __init__(self) -> None:
            self.search = GoogleSearchAPIWrapper(
            google_api_key='',
            google_cse_id='',
        )

    @property
    def name(self) -> str:
        return 'google_search'

    @property
    def description(self) -> str:
        return 'Enter question you would like to know. Returns text answer from the internet.' 

    def execute(self, input: str) -> str:
        
        """Run query through GoogleSearch and parse result."""
        snippets = []
        links = []
        results = self.search._google_search_results(input)
        
        if len(results) == 0:
            return "No good Google Search Result was found"
        for result in results:
            if "snippet" in result:
                snippets.append(result["snippet"])
            if "link" in result:
                links.append(result["link"])

        result_parsed = " ".join(snippets)
        links_parsed = " ".join(links)
        msg = [
            {'role': 'system', 'content': "Answer the user's query based on materials presented, and if possible, the most relevant link which best describes the answer after a "'\n'" then (Source: . Respond as concise as possible. If the given material does not explicitly provide information about user's query, admit so."},
            {'role': 'system', 'content': f"Results Parsed: {result_parsed} and Links Parsed: {links_parsed} \n----\nQUERY: {input}"}
        ]
        response = get_turbo_response(msg)

        return response['content']
