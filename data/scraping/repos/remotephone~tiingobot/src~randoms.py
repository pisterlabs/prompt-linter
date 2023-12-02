import json
import os
import random

import openai
import requests


def get_tax_refund(ssn: str = None):
    if not ssn:
        return "Please provide your real, honest social security number"
    ssn = ssn.replace("-", "")
    try:
        minimum = -int(ssn) * 10
        maximum = int(ssn) * 10
        random.seed()
        refund = random.randint(minimum, maximum)
        print(refund)
        if refund > 0:
            return f"Your refund is ${str(abs(refund))}"
        else:
            return f"You owe ${str(abs(refund))}"
    except ValueError as e:
        return "You owe one million dollars"


def get_artificial_intelligence():
    response = requests.get("https://api.adviceslip.com/advice")
    json_data = json.loads(response.text)
    quote = json_data["slip"]["advice"]
    return quote


def get_artificial_intelligence_v2(question: str) -> str:
    # Use the chatGPT free api to ask a question
    openai.api_key = os.getenv("OPENAI_API_KEY")

    if len(question) > 1000:
        return "Please ask a shorter question"
    # Create a completion object
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": question}])

    # Check if the completion was successful
    if completion.choices[0].message.content is None:
        raise RuntimeError("Unable to get a response from the AI")

    return completion.choices[0].message.content
