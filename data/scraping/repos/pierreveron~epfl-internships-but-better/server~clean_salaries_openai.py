import asyncio
import os
import time

import orjson
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.llms.openai import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import ValidationError

from salaries_types import SalariesDict

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


salary_query = """I have a list of strings representing salaries.
I want to get the salary as a number.
Extract the monthly salary from the string.
Map the string to the number in a json object.
Use the exact strings in the list as the keys of the object.
If there's a distinction between a bachelor and a master salary, pick the master salary.
If more than one number given, always pick the lowest one.
Put null if no salary is found.
Put 0 if the salary is specified as "unpaid".
"""

parser = PydanticOutputParser(pydantic_object=SalariesDict)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\nFormat the following salaries:\n{salaries}\n",
    input_variables=["salaries"],
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
        "query": salary_query,
    },
)


async def clean_salaries(salaries: list[str]):
    """
    Clean a list of salaries using OpenAI.

    Args: salaries (list[str]): A list of salaries.

    Returns: A list of unique salaries in a json format.
    """
    llm = OpenAI(
        model_name="gpt-3.5-turbo-instruct",
        openai_api_key=OPENAI_API_KEY,
        max_tokens=3000,
        request_timeout=60,
    )
    # print(llm)

    print("Number of salaries:", len(salaries))
    # Remove duplicates.
    salaries = list(set(salaries))
    print("Number of unique salaries:", len(salaries))

    total_cost = 0
    total_tokens = 0

    async def async_predict(input_list: list[str]):
        nonlocal total_cost, total_tokens

        _input = prompt.format_prompt(salaries=input_list)
        data: SalariesDict | None = None

        for _ in range(0, 5):
            # Time the request.
            s = time.perf_counter()
            print("Starting request...")

            with get_openai_callback() as cb:
                output = await llm.apredict(_input.to_string())
                total_cost += cb.total_cost
                total_tokens += cb.total_tokens

            elapsed = time.perf_counter() - s
            print(
                f"Request with {len(input_list)} elements took {elapsed:0.2f} seconds."
            )

            try:
                data = parser.parse(output)
            except ValidationError as e:
                print("A validation error occurred:", e)
                continue
            except Exception as e:
                print("An exception occurred:", e)
                continue
            break

        return data

    total_data: SalariesDict = SalariesDict(salaries={})

    s = time.perf_counter()
    while True:
        missing_keys = list(set(salaries) - set(total_data.salaries.keys()))
        if len(missing_keys) == 0:
            break

        # Split the salaries into chunks.
        chunk_size = 50
        chunks = [
            missing_keys[x : x + chunk_size]
            for x in range(0, len(missing_keys), chunk_size)
        ]
        try:
            tasks = [async_predict(chunk) for chunk in chunks]
            data_list = await asyncio.gather(*tasks)

            # Data can be None if there was an error.
            for data in data_list:
                if data is None:
                    continue
                total_data.salaries.update(data.salaries)

        except Exception as e:
            elapsed = time.perf_counter() - s
            print("Total tokens:", total_tokens)
            print(f"Total cost: $", round(total_cost, 2))
            print(f"Total time: {elapsed:0.2f} seconds.")
            print("An error occurred. Please try again.", e)
            raise e

    elapsed = time.perf_counter() - s
    print("Total tokens:", total_tokens)
    print(f"Total cost: $", round(total_cost, 2))
    print(f"Total time: {elapsed:0.2f} seconds.")

    # Send the unique salaries formatted as a json.
    return orjson.loads(total_data.json())
