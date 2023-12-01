import json

import requests

from open_ai_connector.const import OpenAiModels
from open_ai_connector.open_ai_connector import OpenAIConnector
from solver.prompt_builder import prepare_prompt
from solver.solver import Solver

ASSISTANT_CONTENT = """
Based on the question return proper string without any additional info.

###
Use only these categories:
If question is about country return COUNTRY.
If question is about currency return CURRENCY.
If question is about general knowledge return GENERAL.
"""
USER_CONTENT = "{question}"

CURRENCY_API = "http://api.nbp.pl/api/exchangerates/tables/A/"  # USE TABLE A
COUNTRY_API = (
    "https://restcountries.com/v3.1/all?fields=name,population"
)  # USE FIELD POPULATION


def get_currency(oai: OpenAIConnector, question: str):
    response = requests.get(CURRENCY_API)
    data = json.dumps(response.json())
    ASSISTANT_CONTENT = "Based on data return current exchange rate. Data: {data}"
    USER_CONTENT = "{question}"
    prompt = prepare_prompt(
        ASSISTANT_CONTENT.format(data=data), USER_CONTENT.format(question=question)
    )
    answer = oai.generate_answer(model=OpenAiModels.gpt4.value, messages=prompt)
    print(answer)
    return answer


def get_country_info(oai: OpenAIConnector, question: str):
    response = requests.get(COUNTRY_API)
    data = response.json()

    def simplify_data(dict_list):
        simplified_list = []
        for item in dict_list:
            common_name = item["name"]["common"]  # Directly access the common name
            population = item["population"]  # Extract the population
            simplified_list.append({"country": common_name, "population": population})
        return simplified_list

    cleared_data = simplify_data(data)
    print(cleared_data)
    ASSISTANT_CONTENT = "Based only on this data return the number of the population as number, nothing else. Data: {data}"
    USER_CONTENT = "{question}"
    prompt = prepare_prompt(
        ASSISTANT_CONTENT.format(data=cleared_data),
        USER_CONTENT.format(question=question),
    )
    answer = oai.generate_answer(model=OpenAiModels.gpt4.value, messages=prompt)
    print(answer)
    return answer


def general_answer(oai: OpenAIConnector, question: str):
    ASSISTANT_CONTENT = "Answer ultra-briefly to a question."
    USER_CONTENT = "{question}"
    prompt = prepare_prompt(ASSISTANT_CONTENT, USER_CONTENT.format(question=question))
    answer = oai.generate_answer(model=OpenAiModels.gpt4.value, messages=prompt)
    return answer


CATEGORY_TO_FUNCTION_MAP = {
    "CURRENCY": get_currency,
    "COUNTRY": get_country_info,
    "GENERAL": general_answer,
}


def knowledge(input_data: dict) -> dict:
    question = input_data["question"]
    print(input_data)
    print(f"Question: {question}")

    oai = OpenAIConnector()
    prompt = prepare_prompt(ASSISTANT_CONTENT, USER_CONTENT.format(question=question))
    category = oai.generate_answer(model=OpenAiModels.gpt4.value, messages=prompt)
    answer = CATEGORY_TO_FUNCTION_MAP[category](oai, question)
    print(f"Category: {category}")
    prepared_answer = {"answer": answer}
    return prepared_answer


if __name__ == "__main__":
    # TODO: Refactor and use function call
    sol = Solver("knowledge")
    sol.solve(knowledge)
