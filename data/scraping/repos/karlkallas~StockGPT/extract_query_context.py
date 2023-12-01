import os
from typing import Optional
from typing import Tuple

from langchain import OpenAI

ROUTE_DETECTION_PROMPT = f"""
You are given a user query below.
---------------------
{{query}}
---------------------
Extract 1 company name and its ticker from the query in the format NAME-TICKER. If no name or ticker can be extracted then use 0.
Here are some examples:
Query: How is Tesla price looking? its ticker is TSLA.
Answer: Tesla-TSLA

Query: Whats up with Microsoft?
Answer: Microsoft-0
"""


def extract(query: str) -> Optional[Tuple]:
    llm_output = _get_llm_output(query)
    return _format_output(llm_output)


def _get_llm_output(query) -> str:
    try:
        llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
        llm_output = llm(
            ROUTE_DETECTION_PROMPT.format(query=query))
        return llm_output
    except Exception as e:
        print(e)
        return ""


def _format_output(llm_output) -> Optional[Tuple]:
    if llm_output:
        split = llm_output.split("-")
        if len(split) == 2:
            return _clean_string(split[0]), _clean_string(split[1])
    return None


def _clean_string(value: str) -> str:
    return value.replace("\n", "")


if __name__ == '__main__':
    result = extract("How is paypal looking nowadays?")
    print(result)
