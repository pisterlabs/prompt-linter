import argparse
import json
import logging
import os
import time

import openai

from addons.openai_functions import math, memory, wikipedia

logger = logging.getLogger(__name__)
openai.api_key = os.environ["OPENAI_API_KEY"]
GPT3 = "gpt-3.5-turbo-0613"
GPT3_LONG = "gpt-3.5-turbo-16k"
GPT4 = "gpt-4-0613"

FUNCTIONS = [
    math.add,
    math.subtract,
    math.multiply,
    math.divide,
    memory.store_in_memory,
    memory.get_from_memory,
    wikipedia.find_wikipedia_page_key,
    wikipedia.get_wikipedia_page_summary,
    wikipedia.list_wikipedia_sections,
    wikipedia.get_wikipedia_section,
]
FUNCTIONS_BY_NAME = {f.__qualname__: f for f in FUNCTIONS}
FUNCTION_DESCRIPTIONS = []
for f in FUNCTIONS:
    try:
        FUNCTION_DESCRIPTIONS.append(json.loads(f.__doc__))
    except:
        raise ValueError(f"Invalid docstring for {f.__qualname__}")


def run_conversation(user_request: str, use_gpt4: bool) -> None:
    gpt = GPT4 if use_gpt4 else GPT3_LONG
    messages = [
        {
            "role": "user",
            "content": user_request,
        }
    ]

    logger.info("User request: %s", messages[-1])
    keep_talking_to_gpt = True
    while keep_talking_to_gpt:
        try:
            response = openai.ChatCompletion.create(
                model=gpt,
                messages=messages,
                functions=FUNCTION_DESCRIPTIONS,
            )
            response_message = response["choices"][0]["message"]  # type: ignore
            messages.append(response_message)
            logger.info("%s response: %s", gpt, response_message)

            if keep_talking_to_gpt := "function_call" in response_message:
                function_name = response_message["function_call"]["name"]
                fuction_to_call = FUNCTIONS_BY_NAME[function_name]
                function_args = json.loads(
                    response_message["function_call"]["arguments"]
                )
                function_response = fuction_to_call(**function_args)  # type: ignore
                messages.append(
                    {
                        "role": "function",
                        "name": function_name,
                        "content": function_response,
                    }
                )
                logger.info("User response: %s", messages[-1])

        except Exception as exception:  # pylint: disable=broad-exception-caught
            logging.error(exception)
            logging.info("Trying again in 1 second...")
            time.sleep(5)
    logger.info(messages[-1]["content"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=str, required=True)
    parser.add_argument("--gpt4", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_conversation(args.request, args.gpt4)
