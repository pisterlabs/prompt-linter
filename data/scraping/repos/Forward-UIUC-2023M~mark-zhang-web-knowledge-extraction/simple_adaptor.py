"""SimpleAdaptor answers questions with a single attribute answer on a single page."""

import asyncio
from textwrap import dedent
import logging
from dotenv import load_dotenv

import guidance

from utils import flatten, preprocess_html, get_chunks
from framework import Adaptor, PageSourceStrategy

load_dotenv()

# pylint: disable=not-callable


LLM = guidance.llms.OpenAI("gpt-3.5-turbo")


quoting_program = guidance("""
{{#system~}}
You are a verbose and precise quoting program.
{{~/system}}

{{#user~}}
Given a segment of a web page in Markdown format and a question or request \
about the page, you should return a quote from the page that answers the \
question. If there is no relevant quote, you should return "None".
You will read a segment of a web page in Markdown form, then extract the \
relevant segments for the following question:
Question/Request: {{question}}
Here is the segment of the web page:
---
{{context}}
---
Please answer the following question or fulfill the request:
Question/Request: {{question}}
Extract from the given web page the most relevant segments verbatim for the\
answer. Make sure you don't leave out any information. If there are no \
relevant segments, return "None".
{{~/user}}

{{#assistant~}}
{{gen 'answer' temperature=0.0 max_tokens=100}}
{{~/assistant}}
""")

compose_program = guidance("""
{{#system~}}
You are a verbose and precise summarizing program.
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


# limit the number of concurrent calls to the API
CONCURRENCY = 5


async def map_webpage(webpage: str, question) -> str:
    tasks = set()
    results = []

    for chunk in get_chunks(webpage, 1500, 300, seperator="\n"):
        task = quoting_program(
                context=chunk,
                question=question,
                async_mode=True,
                llm=LLM,
                )
        tasks.add(task)
    
        if len(tasks) == CONCURRENCY:
            # wait for any of the tasks to finish
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                results.append(await task)
            
    results.extend(await asyncio.gather(*tasks))
    return [result['answer'] for result in results]


async def qa(html, question, verbose=False):
    html = flatten(html)
    text = preprocess_html(html)

    logging.info("Mapping webpage")   
    quotes = filter(lambda x: x.find("None") == -1,
                    await map_webpage(text, question))

    logging.info("Composing answer")
    output = await compose_program(
        segments=list(quotes), 
        question=question,
        async_mode=True,
        llm=LLM,
        )

    if verbose:
        print(output)

    return output['answer']


class SimpleAdaptor(Adaptor):
    """Answers questions with a single attribute answer on a single page.
    
    SimpleAdaptor uses a map-reduce strategy to answer questions. It first
    quotes relevant segments from each chunk of the page, then composes the
    final answer from the quotes."""

    async def is_suitable(self, question, source_strategy) -> bool:
        is_question_suitable = guidance(dedent("""\
            Is the following question asking for a single answer as supposed \
            to a list of answers? Answer with yes or no.
            Q: {{question}}
            A: {{gen 'answer'}}"""))
        output = await is_question_suitable(question=question, async_mode=True, llm=LLM)
        answer = output['answer'].lower()
        return (isinstance(source_strategy, PageSourceStrategy)
                and answer.find('yes') != -1)

    async def run(self, question, source_strategy):
        return await qa(source_strategy.content, question)
