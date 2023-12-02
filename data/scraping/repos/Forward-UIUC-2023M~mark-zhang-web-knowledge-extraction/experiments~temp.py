
import logging
logging.basicConfig(level=logging.DEBUG)
import asyncio
from dotenv import load_dotenv
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
Given a segment of a web page in Markdown format and a question or request\
about the page, you should return a quote from the page that answers the\
question. If there is no relevant quote, you should return "None".
You will read a segment of a web page in Markdown form, then extract the relevant segments to answer the following question:
Question/Request: {{question}}
Here is the segment of the web page:
---
{{context}}
---
Please answer the following question or fulfill the request:
Question/Request: {{question}}
Extract from the given web page the most relevant segments verbatim for the\
answer. If\ there are no relevant segments, return "None".
{{~/user}}

{{#assistant~}}
{{gen 'answer' temperature=0.1 max_tokens=100}}
{{~/assistant}}
""")

compose_program = guidance("""
{{#system~}}
You are a informative and precise summarizing program.
{{~/system}}

{{#user~}}
Given information extracted from different segments of the webpage, you should\
return a summary of the information that answers the question. If there is no\
relevant information to compose the answer, say so.

Potentially relevant information:
{{#each segments}}
---
{{this}}
{{/each}}

---
Now compose the answer to the following question or fulfill the request:
Question/Request: {{question}}
{{~/user}}

{{#assistant~}}
{{gen 'answer' temperature=0.1 max_tokens=200}}
{{~/assistant}}
""")

FILE = "cs_faculty.html"
QUESTION = """Give me a list of all the faculty members with the title "teaching associate professor"."""

chunk = '[ Sayan Mitra ](/about/people/all-faculty/mitras)\n\nProfessor, Electrical and Computer Engineering\n\n[ _ _ _ _ ](tel:\\(217\\) 333-7824)\n\n[ _ _ _ _ ](mailto:mitras@illinois.edu)\n\n--- \n\n[ Radhika Mittal ](/about/people/all-faculty/radhikam)\n\nAssistant Professor, Electrical and Computer Engineering\n\n[ _ _ _ _ ](tel:)\n\n[ _ _ _ _ ](mailto:radhikam@illinois.edu)\n\n--- \n\n[ Marco Morales Aguirre ](/about/people/all-faculty/moralesa)\n\nTeaching Associate Professor\n\n[ _ _ _ _ ](tel:\\(217\\) 244-8896)\n\n[ _ _ _ _ ](mailto:moralesa@illinois.edu)\n\n--- \n\n[ Klara Nahrstedt ](/about/people/all-faculty/klara)\n\nGrainger Distinguished Chair in Engineering\n\n[ _ _ _ _ ](tel:\\(217\\) 244-6624)\n\n[ _ _ _ _ ](mailto:klara@illinois.edu)\n\n--- \n\n[ Yee Man (Margaret) Ng ](/about/people/all-faculty/ymn)\n\nAssistant Professor, Journalism and Institute of Communications Research\n\n[ _ _ _ _ ](tel:\\(217\\) 300-8186)\n\n[ _ _ _ _ ](mailto:ymn@illinois.edu)\n\n--- \n\n[ David M. Nicol ](/about/people/all-faculty/dmnicol)\n\nProfessor, Electrical and Computer Engineering\n\n[ _ _ _ _ ](tel:\\(217\\) 244-1925)\n\n[ _ _ _ _ ](mailto:dmnicol@illinois.edu)\n\n--- \n\n[ Michael Nowak ](/about/people/all-faculty/mnowak1)\n\nTeaching Assistant Professor\n\n[ _ _ _ _ ](tel:\\(217\\) 244-8894)\n\n[ _ _ _ _ ](mailto:mnowak1@illinois.edu)\n\n--- \n\n[ Idoia Ochoa ](/about/people/all-faculty/idoia)\n\nAssistant Professor, Electrical and Computer Engineering\n\n[ _ _ _ _ ](tel:)\n\n[ _ _ _ _ ](mailto:idoia@illinois.edu)\n\n--- \n\n[ Luke Olson ](/about/people/all-faculty/lukeo)\n\nProfessor and Willett Faculty Scholar\n\n[ _ _ _ _ ](tel:\\(217\\) 244-8422)\n'

output = quoting_program(
    context=chunk,
    question=QUESTION,
    )
print(output)