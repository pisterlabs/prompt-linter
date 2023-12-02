import asyncio
from dotenv import load_dotenv
import logging
load_dotenv()

import guidance
import map_reduce
from utils import flatten, preprocess_html, get_chunks


guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo")

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
Reason: it is unlikely to find a keyword that is an exact match on the page.
Choice: No
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


async def qa(filename, question, verbose=False):
    with open(f"data/html/{filename}", "r", encoding="utf-8") as f:
        html = f.read()
    # text = preprocess_html(html)
    html = flatten(html)

    output = await decide_find(question=question, async_mode=True)
    print(output)
    if output['use_find'].split()[1] == "Choice: No":
        print("Decided not to use find")
        return

    search_term = (output['use_find']
                   .split('\n')[2]
                   .split(":")[1]
                   .strip()
                   .strip('"')
                   .lower())
    print(f'Search Term: {search_term}')
    
    # search for search_term in html and get the segments
    segments = []
    start = 0
    html_lower = html.lower()
    while True:
        index = html_lower.find(search_term, start)
        if index == -1:
            break
        segments.append(html[index-100:index+100])
        start = index + 50

    if len(segments):
        print("Found segments")
        output = await compose_program(segments=segments, question=question)
        print(output)
        return output['answer']

    return await map_reduce.qa(filename, question, verbose=verbose)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    FILE = "cs_courses.html"
    QUESTION = """Which CS courses are data related?"""
    
    async def main():
        print(await qa(FILE, QUESTION, verbose=True))

    asyncio.run(main())
