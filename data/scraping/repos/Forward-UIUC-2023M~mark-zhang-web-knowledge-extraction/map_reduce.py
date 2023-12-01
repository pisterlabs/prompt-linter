import asyncio
from dotenv import load_dotenv
import logging
load_dotenv()

import guidance
from utils import *


guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo")

# pylint: disable=not-callable
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


async def map_webpage(webpage: str, question) -> str:
    tasks = []
    for chunk in get_chunks(webpage, 1500, 300, seperator="\n"):
        task = quoting_program(
            context=chunk,
            question=question,
            async_mode=True
            )
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    return [result['answer'] for result in results]


async def qa(filename, question, verbose=False):
    with open(f"data/html/{filename}", "r", encoding="utf-8") as f:
        html = f.read()
    # text = preprocess_html(html)
    text = flatten(html)

    quotes = filter(lambda x: x.find("None") == -1,
                    await map_webpage(text, question))
    output = await compose_program(segments=list(quotes), question=question)

    if verbose:
        print(output)

    return output['answer']


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    FILE = "cs_faculty.html"
    QUESTION = """Give me a list of all the faculty members with the title "teaching associate professor"."""
    
    async def main():
        print(await qa(FILE, QUESTION, verbose=True))

    asyncio.run(main())
