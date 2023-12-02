from typing import List
import os
from duckduckgo_search import ddg
import openai
from colorama import Fore, Style
from xoxo import Message, SearchResult

openai.api_key = os.environ["OPENAI_API_KEY"]

summary_prompt = """
You are given a query and retrieved search results. Summarize the given search results keeping the user request in mind. 
If you enumerate items, list them on new lines using bullets. Add the source urls in square brackets with the prefix 'url:', for example: [url: the actual url]

RESULTS:
{results}

QUERY: {query}

ANSWER:

"""

class Retriever:
    def __init__(self, k: int = 3) -> None:
        self.k = k

    def trigger(self, query: str, user_request: str = None):
        search_results = self.search(query)
        passages = "\n\n".join([x.get_passage() for x in search_results[: self.k]])
        request, state = (query + "\nUSER QUESTION: " + user_request + "\n", "XOXO") if user_request else (query, "RESULT")
        summary = self.summarize(request, passages)
        return Message(state, summary)

    def search(self, query: str) -> List[SearchResult]:
        # todo: import region from config
        try:
            response = ddg(query, region="us-en", safesearch="off", max_results=self.k)
            return [
                SearchResult(x["title"], x["href"], x["body"]) for x in response
            ]

        except Exception as e:
            print(e)
            return []

    def summarize(self, query: str, context: str) -> str:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=summary_prompt.format(query=query, results=context),
            temperature=0.5,
            max_tokens=350,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].text.strip()

    @staticmethod
    def format_retriever_msg(s: str) -> str:
        prefix = f"{Fore.CYAN} ~~ response:{Style.RESET_ALL}"
        return prefix + "\n" + f"{Fore.CYAN}{s}{Style.RESET_ALL}" + "\n" + f"{Fore.CYAN} {'~'*30}"