from typing import Optional, Tuple
from langchain.llms.base import BaseLLM
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

from web_extractor import WebExtractor
from bing_search import BingSearch

class WebSearcher:
    TEMPLATE = """You are a query answering bot. Given the following Bing search, and top 3 results.
You can choose one of the following tools:
1. Answer["Your answer"]: Provide the answer based on the search results.
2. Browse[]: Open up one of the web results and look at some of the content

NOTE: If you don't know the exact answer, or you're unsure, it's always better to browse.

Example 1:
Search query: How many episodes of The Office are there?
Result snippets:
WEB RESULT
List of The Office (American TV series) episodes - Wikipedia
A total of 201 episodes of The Office aired over nine seasons. The first set of webisodes, titled The Accountants, consisted of ten episodes and ran between the second and third seasons...
[snip]
Thought: I can see the answer from the snippet
Tool:
Answer["Based on the Bing search snippet, there are 201 episodes of The Office."]

Example 2:
Search query: Picard season 3 episode count
Result snippets:
WEB RESULT
Star Trek: Picard (season 3) - Wikipedia
The third and final season of the American television series Star Trek: Picard features the character Jean-Luc Picard in the year 2401 as he reunites with the former command crew of the USS Enterprise (Geordi La Forge, Worf, William Riker, Beverly Crusher, and Deanna Troi) while facing a mysterious …
[snip]
Thought: I still don't see the answer to the question
Tool:
Browse[]

Your turn!
Search query: {search_query}
Result snippets:
{snippets}
Thought:"""

    BROWSE_TEMPLATE = """Extract as much relevant information you can from a bing search result snippet, and an (attempt) to load one of the results.
You may encounter very incomplete snippets, or "browsed text" entries that say "403" or some other web error. Do your best to extract anything relevant to the search.

Example 1:
Search query: Picard season 3 episode count
Result snippets:
WEB RESULT
Star Trek: Picard (season 3) - Wikipedia
The third and final season of the American television series Star Trek: Picard features the character Jean-Luc Picard in the year 2401 as he reunites with the former command crew of the USS Enterprise (Geordi La Forge, Worf, William Riker, Beverly Crusher, and Deanna Troi) while facing a mysterious …
[snip]
Browsed text:
URL: https://en.wikipedia.org/wiki/Star_Trek:_Picard_(season_3)
The season premiered on the streaming service Paramount+ on February 16, 2023, and is running for 10 episodes until April 20.
Answer: There are 10 episodes in season 3 of Star Trek: Picard.

Example 2:
Search query: What is Wobblefuzzle Twiddle?
Result snippets:
WEB RESULT
Thumb twiddling - Wikipedia
Thumb twiddling is an activity that is done with the hands of an individual whereby the fingers are interlocked and the thumbs circle ar...
[snip]
Browsed text:
URL: https://en.wikipedia.org/wiki/Thumb_twiddling
Thumb twiddling is an activity that is done with the hands of an individual whereby the fingers are interlocked and the thumbs circle ar...
Answer: I can see results for thumb twiddling, but it's still unclear what a Wobblefuzzle Twiddle is.

Your turn!
Search query: {search_query}
Result snippets:
{snippets}
Browsed text:
URL: {browsed_url}
{extracted_text}
Answer:"""
    def __init__(self) -> None:
        self.web_extractor = WebExtractor()
        self.web_searcher = BingSearch()
        self.llm = ChatOpenAI(temperature=0.0) # type: ignore

    async def run(self, search_query: str) -> str:
        results = await self.web_searcher.results(search_query)
        if len(results) == 0:
            return ""

        snippets = ['WEB RESULT\n' + result.name + '\n' + result.snippet + '\n\n' for result in results]

        url_to_extract = results[0].url
        chunks = await self.web_extractor.extract_text(url_to_extract)
        if len(chunks) == 0:
            chunks = ["There was an error loading the web page"]

        chain = LLMChain(llm=self.llm, prompt=PromptTemplate(
                template=self.BROWSE_TEMPLATE,
                input_variables=["snippets", "search_query", "extracted_text", "browsed_url"]
            )
        )
        print(snippets)
        print(chunks[0])
        response = await chain.arun(
            search_query=search_query,
            snippets=snippets,
            extracted_text=chunks[0],
            browsed_url=url_to_extract
        )
        print(response)
        return response.strip()
