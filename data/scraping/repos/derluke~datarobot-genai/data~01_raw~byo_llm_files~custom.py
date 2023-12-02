# type: ignore
import cohere
import pandas as pd
from datarobot_drum import RuntimeParameters


def load_model(*args, **kwargs):
    cohere_api_key = RuntimeParameters.get("cohere_api_token")["apiToken"]
    return cohere.Client(cohere_api_key)


def score(data, model, **kwargs):
    prompts = data["promptText"].tolist()
    responses = []

    for prompt in prompts:
        response = model.generate(prompt)
        responses.append(response[0].text)

    return pd.DataFrame({"resultText": responses})
