import os

import wolframalpha
from openai_util.function_call.openaifunc_decorator import openai_func

wolframalpha_id = os.getenv('WOLFRAMALPHA_ID')


@openai_func
def query_wolframalpha(query: str) -> str:
    """
    Run query through WolframAlpha and parse result.

    :param query: query string
    """
    res = wolframalpha.Client(wolframalpha_id).query(query)

    try:
        assumption = next(res.pods).text
        answer = next(res.results).text
    except StopIteration:
        return "Wolfram Alpha wasn't able to answer it"

    if answer is None or answer == "":
        # We don't want to return the assumption alone if answer is empty
        return "No good Wolfram Alpha Result was found"
    else:
        return f"Assumption: {assumption} \nAnswer: {answer}"

# print(query_wolframalpha("what is 2x+18=x+5?"))
# print(query_wolframalpha("What is the capital of China?"))
