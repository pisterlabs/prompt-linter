"""Web search adaptor"""
# pylint: disable=not-callable

import logging
from textwrap import dedent
from dotenv import load_dotenv

from googlesearch import search
import guidance

from utils import fetch_html_async
from simple_adaptor import SimpleAdaptor
from framework import Adaptor, SearchSourceStrategy, PageSourceStrategy

load_dotenv()

LLM = guidance.llms.OpenAI("gpt-3.5-turbo")


class GoogleSearchAdaptor(Adaptor):
    """Web search adaptor"""
    def __init__(self):
        pass

    async def is_suitable(self, question, source_strategy) -> bool:
        """Check if the adaptor is suitable for the given question and source strategy."""
        return isinstance(source_strategy, SearchSourceStrategy)

    async def run(self, question, source_strategy) -> str:
        """Run the task."""
        logging.info("Running web search adaptor")
        logging.info("Running Google search")
        results = search(query=question, num=10, stop=10, pause=2.0)
        logging.info("Google search finished")
        for res in results:
            logging.info("Fetching %s", res)
            html = await fetch_html_async(res)
            if html is None or len(html) <= 0:
                logging.info("%s contains no information", res)
                continue
            logging.info("Running SimpleAdaptor from page %s", res)
            source = PageSourceStrategy(url=res, content=html)
            answer = await SimpleAdaptor().run(question, source)
            if await self.check_has_answer(answer, question):
                return answer
        return None

    async def check_has_answer(self, answer, question):
        """Check if question is answered"""
        prompt = guidance(dedent("""\
            {{#system~}}
            You are a helpful agent
            {{~/system}}

            {{#user~}}
            {{answer}}
            Does the answer find relevant information to answer the question: {{question}}? Answer with Yes or No.
            {{~/user}}

            {{#assistant~}}
            {{gen 'result' temperature=0.0 max_tokens=2}}
            {{~/assistant}}"""))
        output = await prompt(answer=answer, question=question, 
                              async_mode=True, llm=LLM)
        print(output)
        if output['result'].lower().find("yes") >= 0:
            return True
        return False

