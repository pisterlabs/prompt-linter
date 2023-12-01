import openai
from defines import (
    AI_MODEL,
    INSTRUCT_GPT_PROMPT_ANSWER_OPENING,
    INSTRUCT_GPT_PROMPT_HEADER,
)


def request_response_from_ai_model(prompt):
    """Tries to get a response from GPT

    Args:
        prompt (str): the prompt that will be sent to GPT

    Returns:
        str: either a valid response or None
    """
    # Read API key from file
    with open("api_key.txt", "r", encoding="utf8") as file:
        openai.api_key = file.read().strip()

    prompt = INSTRUCT_GPT_PROMPT_HEADER + prompt + INSTRUCT_GPT_PROMPT_ANSWER_OPENING

    response = openai.ChatCompletion.create(
        model=AI_MODEL,
        temperature=1,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )

    return response.choices[0]["message"]["content"]
