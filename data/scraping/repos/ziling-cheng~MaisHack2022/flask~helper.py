import yaml
import cohere
from pyprojroot import here


def get_api_key(secrets_yaml):
    """
    Returns the api key from a given secrets file
    """
    with open(secrets_yaml, "r") as f:
        secrets = yaml.load(f, Loader=yaml.FullLoader)

    return secrets["cohere"]["api_key"]


def request_generation(prompt, key=None, **kwargs):

    """
    Requests text generation via Co:here API

    Relevant here: https://docs.cohere.ai/generate-reference/
    """

    if key is None:
        key = get_api_key(here("secrets.yaml"))

    co = cohere.Client(f"{key}")
    response = co.generate(prompt=prompt, **kwargs)

    # return response.generations[0].text
    return response.generations
