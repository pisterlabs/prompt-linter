import openai
import os
from metaphor_python import Metaphor
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re

from .prompts import *

load_dotenv()

# Load in API keys from OpenAI and Metaphor
openai.api_key = os.getenv("OPENAI_API_KEY")
metaphor = Metaphor(os.getenv("METAPHOR_API_KEY"))


def extract_bullet_points(text):
    """
        Given a string of text, extract data from just the bullet points and return each bullet as an item in a list.
        :param text: A paragraph/string of text that has some bullet points in it.
        :return: List of bullet points. Returns an empty list if no bullet points are found.
    """
    # Regular expression pattern to identify bullet points
    bullet_point_pattern = re.compile(r'(?:\d+\.)?\s*[-*+]\s*(.*)|^\d+\.\s*(.*)', re.MULTILINE)

    # Find all matches in the input text
    matches = bullet_point_pattern.findall(text)

    # Extract the matched bullet points, taking non-empty matching group
    bullet_points = [match[0] if match[1] == '' else match[1] for match in matches]

    return bullet_points


def truncate_to_tokens(input_string, max_tokens=250):
    """
    Truncate the input string to keep only the first `max_tokens` tokens.

    :param input_string: The input text to be truncated.
    :param max_tokens: The maximum number of tokens to keep.
    :return: Truncated string.
    """
    tokens = input_string.split()
    truncated_tokens = tokens[:max_tokens]
    truncated_string = ' '.join(truncated_tokens)
    return truncated_string


def fetch_metaphor_results(inputs):
    """
    Given a dictionary of inputs, return a list of results from Metaphor. Returned list will be a list of
    "DocumentContent" objects. This will include title, id, content, and url.
    :param inputs:
    :return:
    """
    # Get suggestions directly from Metaphor
    prompt = metaphor_direct_suggestions_prompt(inputs)
    response = metaphor.search(
        query=prompt,
        use_autoprompt=False,
        start_published_date="2019-09-01",
        num_results=5,
        type='neural'
    )
    # Get content for each result
    contents = metaphor.get_contents(ids=[result.id for result in response.results])

    return contents.contents


def fetch_suggestions_from_gpt(inputs):
    """
    Given a dictionary of inputs, return a list of results from GPT-3. Returned list will be a list of
    :param inputs:
    :return:
    """
    # Get suggestions from GPT-3
    prompt_for_gpt = gpt_direct_suggestions_prompt(inputs)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": gpt_direct_suggestions_system_message
            },
            {
                "role": "user",
                "content": prompt_for_gpt
            }
        ],
        temperature=1,
        max_tokens=405,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Extract bullet points from GPT-3 response and for each bullet point, search Metaphor for a localized conrete
    # suggestion
    return_list = []
    for item in extract_bullet_points(response["choices"][0]["message"]["content"]):
        # Search metaphor - get only one result (saves some content
        links_to_recommendations = metaphor.search(
            query=item + "in " + inputs["location"],
            use_autoprompt=True,
            num_results=1,
            type='neural'
        )
        # Get content for each result
        content_for_link = metaphor.get_contents(ids=[result.id for result in links_to_recommendations.results])
        return_list += content_for_link.contents

    return return_list


def summarize_each_suggestion(list_of_suggestions):
    """
    Given a list of suggestions (DocumentContent objects), summarize each suggestion's content using GPT-3.
    Then return the same list of suggestions with the content replaced with the summarized content.
    :param list_of_suggestions:
    :return:
    """
    # Create a single summary of contents to feed into a single prompt
    summary_contents = ""
    i = 1
    for content in list_of_suggestions:
        content.extract = truncate_to_tokens(BeautifulSoup(content.extract, "html.parser").
                                             get_text(separator=" ", strip=True))
        summary_contents += f"{str(i)}. {content.title} + content.extract\n"
        i += 1

    # Prompt GPT to summarize each suggestion's content
    gpt_excerpts_summarization_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "system",
                "content": gpt_rewrite_excerpt_system_message
            },
            {
                "role": "user",
                "content": summary_contents
            }
        ],
        temperature=1,
        max_tokens=8096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # Extract summarized suggestion contents from GPT response
    list_of_summarized_excerpts = extract_bullet_points(
        gpt_excerpts_summarization_response["choices"][0]["message"]["content"])

    # Split all bullet points into their own list item and assign to each suggestion
    for i in range(len(list_of_summarized_excerpts)):
        list_of_suggestions[i].extract = list_of_summarized_excerpts[i]

    return list_of_suggestions


def summarize_all_suggestions(list_of_suggestions, inputs):
    """
    Given a list of suggestions (DocumentContent objects), summarize all suggestions into a single summary using GPT-3.
    Prompt is informed by user inputs.
    :param list_of_suggestions:
    :param inputs:
    :return:
    """
    summarization_prompt = ""
    i = 1
    # Create a single prompt from all the DocumentContent objects
    # We limit the number of words from each extract due to context window (token) limits from GPT.
    for suggestion in list_of_suggestions:
        summarization_prompt += str(i) + ". Title: " + suggestion.title + " Extract: " + suggestion.extract + "\n"
        i += 1

    final_prompt = gpt_output_recommendations_prompt(inputs, summarization_prompt)
    # Call GPT to summarize
    summary_of_activities = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": gpt_summarize_activities_system_message
            },
            {
                "role": "user",
                "content": final_prompt
            }
        ],
        temperature=1,
        max_tokens=2096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return summary_of_activities["choices"][0]["message"]["content"]


def execute_searching_workflow(inputs):
    """
    Executes the whole workflow of searching for suggestions. Returns a tuple of the list of suggestions and the
    overall summary of the suggestions. NOTE: The returned list will be DocumentContents objects
    :param inputs:
    :return:
    """
    # Get suggestions directly from Metaphor (returns a list of DocumentContents)
    suggestion_results = fetch_metaphor_results(inputs)
    # Append suggestions from GPT-3 that're informed by Metaphor's search results (returns a list of DocumentContents)
    suggestion_results += fetch_suggestions_from_gpt(inputs)
    # Summarize each suggestion's content using GPT-3 and update DocumentContents list
    suggestion_results = summarize_each_suggestion(suggestion_results)
    # Summarize all suggestions into a single summary
    overall_summary = summarize_all_suggestions(suggestion_results, inputs)

    return suggestion_results, overall_summary
