import re

import tiktoken  # type: ignore
import openai  # type: ignore

from api.constants import SMALL_GPT_MODEL, SMALL_TOKENS, LARGE_GPT_MODEL, LARGE_TOKENS, PERSONALITY, OPEN_AI_API_KEY, SUMMARY_PERSONALITY

# Set up your OpenAI API key

client = openai.OpenAI(api_key=OPEN_AI_API_KEY)


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def evaluate_module(code: str):
    """Analyze the module using GPT-3.5-turbo and return an analysis."""
    max_prompt_tokens = LARGE_TOKENS
    max_tokens = SMALL_TOKENS

    num_tokens = num_tokens_from_string(code)

    if num_tokens >= max_prompt_tokens:
        return "The code is too long to analyze."

    try:
        response = client.chat.completions.create(
            model=LARGE_GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": PERSONALITY,
                },
                {
                    "role": "user",
                    "content": f"Review the following code: \n{code}",
                },
            ],
            max_tokens=max_tokens,
            n=1,
            temperature=0.4,
        )

    except openai.APIError as err:
        print(f"OpenAI API Error: {err}")
        return "OpenAI API Error"

    analysis = response.choices[0].message["content"].strip()
    return analysis


def evaluate_function(code: str):
    """Analyze the function using GPT-4 and return an analysis."""
    max_prompt_tokens = SMALL_TOKENS
    max_tokens = SMALL_TOKENS

    num_tokens = num_tokens_from_string(code)

    if num_tokens >= max_prompt_tokens:
        return "The code is too long to analyze."

    try:
        response = client.chat.completions.create(
            model=SMALL_GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": PERSONALITY,
                },
                {
                    "role": "user",
                    "content": f"Review the following code: {code}",
                },
            ],
            max_tokens=max_tokens,
            n=1,
            temperature=0.4,
        )

    except openai.APIError as err:
        print(f"OpenAI API Error: {err}")
        return "OpenAI API Error"

    analysis = response.choices[0].message["content"].strip()
    return analysis


def evaluate_summary(analysis):
    """Create a summary for the GitHub issue based on the analysis."""
    try:
        response = openai.ChatCompletion.create(
            model=SMALL_GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SUMMARY_PERSONALITY,
                },
                {
                    "role": "user",
                    "content": f"Based on the following code analysis, provide a summary to create a GitHub issue with an 'Issue Title' and a 'Description':\n\n{analysis}\n\n Please specify if the issue is a 'defect' or an 'improvement'. Do not provide any other information.",
                },
            ],
            max_tokens=7000,
            n=1,
            temperature=0.4,
        )
    except openai.APIError as err:
        print(f"OpenAI API Error: {err}")
        return "", "OpenAI API Problem"

    message_content = response["choices"][0]["message"]["content"].encode("utf-8").decode("utf-8")
    title_search = re.search(r"Issue Title: (.+)", message_content)
    description_search = re.search(r"Description: (.+)", message_content)

    title = title_search.group(1) if title_search else ""
    description = description_search.group(1) if description_search else ""

    return title, description  # Simply return the title and description
