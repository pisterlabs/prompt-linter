import json
import os
from typing import Any

import pinecone
from openai import OpenAI

TEXT_MODEL: str = "gpt-3.5-turbo-0301"
# TEXT_MODEL: str = "gpt-4-1106-preview"

with open("keys.txt", "r") as key_file:
    api_keys = [key.strip() for key in key_file.readlines()]
    client = OpenAI(api_key=api_keys[0])
    pinecone.init(
        api_key=api_keys[1],
        environment=api_keys[2],
    )


def fact_rephrase(phrase: str) -> list[str]:
    """
    Given a sentence, break it up into individual facts.
    :param phrase: A phrase containing multiple facts to be distilled into separate ones
    :return: The split-up factual statements
    """
    msgs: list[dict] = [
        {
            "role": "system",
            "content": "You are a writing assistant. "
            "Help me split up the sentences I provide you into facts. "
            "Each fact should be able to stand on it's own."
            "Tell me each fact on a new line, "
            "do not include anything in your response other than the facts.",
        }
    ]
    prompt: str = f"Split this phrase into facts: {phrase}"
    msgs.append({"role": "user", "content": prompt})  # build current history of conversation for model

    res: Any = client.chat.completions.create(model=TEXT_MODEL, messages=msgs, temperature=0)  # conversation with LLM
    facts: str = str(res.choices[0].message.content).strip()  # get model response
    return [fact.strip() for fact in facts.split("\n")]


if __name__ == "__main__":
    print(
        fact_rephrase(
            "Caleb Brown is married to Evelyn Stone-Brown, the local blacksmith, and they live in a cozy house in Ashbourne."
        )
    )
