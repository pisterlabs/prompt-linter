import openai

from api.config import settings


def get_openai() -> openai:
    if settings.OPENAI_API_KEY is None:
        raise ValueError(
            "OPENAI_API_KEY cannot be found from the environment variable."
            "Please read the detailed instructions at README.md"
        )
    openai.api_key = settings.OPENAI_API_KEY
    yield openai
