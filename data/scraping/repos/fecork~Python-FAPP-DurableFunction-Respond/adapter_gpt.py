import logging
import os
import sys
import openai

import numpy as np
from typing import Dict

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)
from Utilities.load_parameter import load_parameters


def login_openai() -> Dict:
    """
    This is a function for  login to openai.
    """
    logging.warning("Executing login_openai")
    openai_credentials = os.environ["OPENAIKEY"]
    if openai_credentials:
        openai.api_key = openai_credentials
        return openai

    else:
        print("No credentials for openai")


def ask_openai(text: str, task: str) -> dict:
    """
    This is a function for
    ask question to GPT API by OpenAI.
    """
    logging.warning("Executing ask_openai")

    loaded_parameters = load_parameters()
    if task == "cancellation":
        parameters = loaded_parameters["open_ai_parameters"]

    if task == "change":
        parameters = loaded_parameters["open_ai_parameters_change"]

    if task == "classification":
        parameters = loaded_parameters["open_ai_parameters_classification"]

    if task == "list":
        parameters = loaded_parameters["open_ai_parameters_list"]

    openai = login_openai()
    prompt = parameters["prompt"]
    prompt = f"{prompt}:\n\n{text}"
    response = openai.Completion.create(
        engine=parameters["model"],
        prompt=prompt,
        temperature=parameters["temperature"],
        max_tokens=parameters["max_tokens"],
        top_p=parameters["top_p"],
        frequency_penalty=parameters["frequency_penalty"],
        presence_penalty=parameters["presence_penalty"],
        logprobs=1,
    )

    response_mean_probability = mean_probability(response)

    return {
        "text": response.choices[0].text.lstrip(),
        "meanProbability": response_mean_probability,
    }


def mean_probability(response: object) -> float:
    """
    This is a function for
    calculate mean probability.
    """
    logging.warning("Executing mean_probability")
    list_probs = []
    list_top_logprobs = list(response.choices[0].logprobs.top_logprobs)
    for top_logprobs_object in list_top_logprobs:
        for key, logprob in top_logprobs_object.items():
            list_probs.append(np.e ** float(logprob))

    mean_probability = sum(list_probs) / len(list_probs) * 100

    return mean_probability
