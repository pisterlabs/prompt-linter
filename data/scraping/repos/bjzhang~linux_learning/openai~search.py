#!/usr/bin/env python3
import os
import sys
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from openai import OpenAI

debug=False
verbose=False

question = sys.argv[1]

search = GoogleSearchAPIWrapper()

def top5_results(query):
    return search.results(query, 5)

client = OpenAI()
if debug:
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for computer architecture and operating system"},
            {"role": "user", "content": question}
        ]
    )

    #ePrint the entire response
    print(response)

if debug:
    print("with assistant")

tool = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=top5_results
)

result = tool.run(question)

if verbose:
    print(result)

question_with_prompt='''
You are a knowledgeable and helpful person that can answer any questions. Your task is to answer the following question delimited by triple backticks. Keep output as same as input language. Do necessary translation if needed.

Question:
```
''' + question + '''
```
It's possible that the question, or just a portion of it, requires relevant information from the internet to give a satisfactory answer. The relevant search results provided below, delimited by triple quotes, are the necessary information already obtained from the internet. The search results set the context for addressing the question, so you don't need to access the internet to answer the question.

Write a comprehensive answer to the question in the best way you can. If necessary, use the provided search results.

For your reference, today's date is 2023-11-13 16:10:27.

---

If you use any of the search results in your answer, always cite the sources at the end of the corresponding line, similar to how Wikipedia.org cites information. Use the citation format [[NUMBER](URL)], where both the NUMBER and URL correspond to the provided search results below, delimited by triple quotes.

Present the answer in a clear format.
Use a numbered list if it clarifies things
Make the answer as short as possible, ideally no more than 150 words.
'''

merged_result = ''
# Format and print the data
for entry in result:
    if debug:
        print(entry)
    formatted_output = f"URL: {entry['link']}\nTITLE: {entry['title']}\nCONTENT: {entry['snippet']}\n"
    merged_result += formatted_output

if debug:
	print(question_with_prompt)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant for computer architecture and operating system"},
        {"role": "assistant", "content": merged_result},
        {"role": "user", "content": question_with_prompt}
    ]
)

# Print the entire response
if debug:
	print(response)

# Extract real content from the response
real_content = response.choices[0].message.content

# Output the real content
print(real_content)
