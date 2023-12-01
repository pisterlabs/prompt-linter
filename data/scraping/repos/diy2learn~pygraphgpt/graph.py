import os
import openai

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_base = os.getenv("OPENAI_API_URL")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_DEPLOYMENT_NAME = os.getenv("OPENAI_MODEL_DEPLOYMENT_NAME")


def find_relations(input_text: str) -> str:
    system_msg = f"""
    Given a prompt, extrapolate entities and their relationships as possible from it and provide a list of updates.
    If an update is a relationship, provide [ENTITY 1, RELATIONSHIP, ENTITY 2]. The relationship is directed, so the order matters.
    Entity must only be a generic term or name and can not be empty.

    Example:
    prompt:
    ///
    Luke and Leia: Siblings
    Han and Chewie: Partners, Bffs. Han is also the love interest, and later husband of Leia. Friends with Luke.
    ///
    updates:
    [
        ['Luke', 'Sibling', 'Leia'],
        ['Han', 'Partner', 'Chewie'],
        ['Han', 'Husband', 'Leia'],
        ['Han', 'Friend', 'Luke'],
    ]
    """

    user_msg = f"""
    prompt: ///
    {input_text}
    ///
    updates:
    """
    response = openai.ChatCompletion.create(
    engine=OPENAI_MODEL_DEPLOYMENT_NAME,
    messages = [{"role":"system","content":system_msg},
                {"role":"user","content":user_msg}],
    temperature=0,
    max_tokens=256,
    )

    return response["choices"][0]["message"]["content"]
