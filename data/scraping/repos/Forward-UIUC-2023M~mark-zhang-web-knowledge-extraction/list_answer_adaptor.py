"""Adaptor for list answer questions."""

import logging
from dotenv import load_dotenv
import guidance
from framework import Adaptor, PageSourceStrategy

load_dotenv()

LLM = guidance.llms.OpenAI("gpt-3.5-turbo")

# pylint: disable=not-callable
decide_find = guidance("""
{{#system~}}
You are a intelligent and resourceful web information retreival agent planner.
{{~/system}}

{{#user~}}
Given a question, you should decide whether to use the find command to find \
the relevant information from the web page. If you decide to use the find \
command, you should also decide what search term to use. Make sure to use \
short keywards as longer phrases are less likely to be found on the page. \
{{~/user}}

{{#assistant~}}
Reason: there might not be an exact match for the price on the page.
Choice: No
{{~/assistant}}

{{#user~}}
Question/Request: Which laptops are under $1000?
Use the find command to find the relevant information from the web page?
{{~/user}}

{{#assistant~}}
Reason: there might not be an exact match for the price on the page.
Choice: No
{{~/assistant}}

{{#user~}}
Question/Request: Where is the opening ceremony of SXSW taking place?
Use the find command to find the relevant information from the web page?
{{~/user}}

{{#assistant~}}
Reason: Easy to find an exact match for opening ceremony.
Choice: Yes
Search Term: "opening"
{{~/assistant}}

{{#user~}}
Question/Request: Which camera are made by "Sony"?
Use the find command to find the relevant information from the web page?
{{~/user}}

{{#assistant~}}
Reason: an exact match for the brand is necessary to accurately answer the \
question. And since the brand is a single word, it is likely to find an exact \
match.
Choice: Yes
Search Term: "Sony"
{{~/assistant}}

{{#user~}}
Question/Request: {{question}}
Use the find command to find the relevant information from the web page?
{{~/user}}

{{#assistant~}}
{{gen 'use_find'}}
{{~/assistant}}
""")

compose_program = guidance("""
{{#system~}}
You are a verbose and precise summarization assistant.
{{~/system}}

{{#user~}}
Given information extracted from different segments of the webpage, you should \
return a summary of the information that answers the question. Critically and \
meticulously analyze every piece of information to not leave out relevant \
information or include irrelevant information in the final answer. Notice \
different segments might contain partially overlapping information. In that \
case, Make sure not to leave anything out when question is examining a \
between entities. If there is no relevant information to compose the answer, \
say so.

Potentially relevant information:
{{#each segments}}---
{{this}}
{{/each}}

---
Now compose the answer to the following question or fulfill the request:
Question/Request: {{question}}
{{~/user}}

{{#assistant~}}
{{gen 'answer' temperature=0.0 max_tokens=300}}
{{~/assistant}}
""")


class ListAnswerAdaptor(Adaptor):
    """Adaptor for list answer questions."""

    def __init__(self):
        self.search_term = None

    async def is_suitable(self, question, source_strategy) -> bool:
        output = await decide_find(question=question, async_mode=True, llm=LLM)
        if output['use_find'].split()[1] == "Choice: No":
            logging.info("Decided not to use find")
            return False

        logging.info(output)

        search_term = (output['use_find']
                    .split('\n')[2]
                    .split(":")[1]
                    .strip()
                    .strip('"')
                    .lower())
        logging.info('Search Term: %s', search_term)
        self.search_term = search_term

        return isinstance(source_strategy, PageSourceStrategy)

    async def run(self, question, source_strategy):
        assert self.search_term, "Search term not set"

        # search for search_term in html and get the segments
        html = source_strategy.content
        segments = []
        start = 0
        html_lower = html.lower()
        while True:
            index = html_lower.find(self.search_term, start)
            if index == -1:
                break
            segments.append(html[index-150:index+150])
            start = index + 50

        if len(segments) <= 0:
            logging.warning("No relevant information found")
            return

        output = await compose_program(
            segments=segments, question=question, async_mode=True, llm=LLM)
        print(output)
        return output['answer']
