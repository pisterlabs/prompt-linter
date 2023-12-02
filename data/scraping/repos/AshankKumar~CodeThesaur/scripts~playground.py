from metaphor_python import Metaphor
from dotenv import load_dotenv
from githubClass import projectList
import os
import openai
import tiktoken
import pdb

load_dotenv()

metaphor = Metaphor(os.getenv("METAPHOR_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")
ENCODER = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
MAX_TOKENS = 4097 - 150 - 1000 # dumb way to account for function and output. TODO fix later
SYSTEM_PROMPT = '''
You are assisting in translating user queries into optimized queries for the Metaphor API, which is designed to retrieve links from the internet based on how people typically describe and share them. Here's how you should format and enhance the queries:
Avoid Keyword Searches: Instead of plain keywords, try to frame the query like someone describing or sharing a link on the internet. For instance, instead of "Jeopardy archive", you'd want "Here is the Jeopardy archive:".
Rephrase Questions as Answers: Users may input queries as questions, but questions are not the most effective prompts for this model. Instead, transform these questions into statements that look like answers. For example, if someone asks "What's the best tutorial for baking?", you should convert it to "This is the best tutorial for baking:".
Use Descriptive Modifiers: If the original query hints at a specific type or style of result, incorporate that information. If a user is looking for something humorous or a specific platform link like Goodreads, ensure it's reflected in the modified query.
End with a Colon: Many of the effective prompts for the Metaphor API end with colons, imitating the way people introduce links. Make sure your transformed query also ends with a colon.
Given this guidance, your task is to take a user query, such as "projects similar to dotenv in python", and transform it into an optimized query for the Metaphor API, like "Here are some projects similar to dotenv in python:".
'''
EXTRACTION_PROMPT = "Consider the data below and segment it into multiple githubProject structures: '\n%s'"

EXTRACTION_TOKENS = len(ENCODER.encode(str(EXTRACTION_PROMPT)))
FUNCTION_TOKENS = len(ENCODER.encode(str(projectList.openai_schema)))
OUTPUT_TOKENS = 1000
TOKENS_PER_MESSAGE = 4097 - FUNCTION_TOKENS - EXTRACTION_TOKENS - OUTPUT_TOKENS

def get_query_from_gpt(user_query):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]
    )

    return completion["choices"][0]["message"]["content"]

def get_metaphor_results(query):    
    search_response = metaphor.search(
    query,
    include_domains=["github.com"],
    )

    contents_response = search_response.get_contents()
    return [f"Title: {content.title}\nURL: {content.url}\nContent:\n{content.extract}\n" for content in contents_response.contents]

def extract_details(metaphor_response):
    output = projectList(projects=[])


    idx = 0
    while idx < len(metaphor_response):

        '''
        load curr idx and set tokens and contents

        while more contents remain and adding those tokens would not exceed our limit do so

        when can add no longer execute and loop back
        '''
        curr_tokens = 0
        curr_contents = []
        while idx < len(metaphor_response) and curr_tokens+(tokens :=  len(ENCODER.encode(metaphor_response[idx]))) < MAX_TOKENS:
            curr_tokens += tokens
            curr_contents.append(metaphor_response[idx])
            idx += 1 

        print(f'{idx} {curr_tokens}')
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            temperature=0.1,
            functions=[projectList.openai_schema],
            function_call={"name": projectList.openai_schema["name"]},
            messages=[
                {
                    "role": "user",
                    "content": EXTRACTION_PROMPT % curr_contents,
                },
            ],
            max_tokens=OUTPUT_TOKENS,
        )

        curr_contents = []
        curr_tokens = 0
        extracted_projects = projectList.from_response(completion)
        output.projects.extend(extracted_projects.projects)


    return output

if __name__ == "__main__":
    # while True:
    user_query = 'I need a project similar to dotenv in python'
    gpt_query = get_query_from_gpt(user_query)
    print(f"GPT Formatted QUERY:{gpt_query}")
    contents = get_metaphor_results(gpt_query)
    print(extract_details(contents))

