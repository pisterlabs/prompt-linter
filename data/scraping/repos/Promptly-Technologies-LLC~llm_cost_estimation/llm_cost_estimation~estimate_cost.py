from typing import Union
from .models import models
from .count_tokens import count_tokens


# Define a function that estimates the cost of a prompt + completion for a given prompt
# (string) or chat prompt (list of dicts) for a particular LLM model
def estimate_cost(prompt: Union[str, list[dict]], model: str) -> float:
    """
    Calculates the cost of requesting a completion for a given LLM.
    """

    # Count the number of tokens in the text
    token_count_prompt, token_count_completion = count_tokens(prompt, model)

    # Get the respective model's cost per token
    for mdl in models:
        if mdl["name"] == model:
            cost_prompt = token_count_prompt * eval(mdl["prompt_cost_per_token"])
            cost_completion = token_count_completion * eval(
                mdl["completion_cost_per_token"]
            )

    # Calculate the total cost of encoding the text
    cost = cost_prompt + cost_completion

    return cost


if __name__ == "__main__":
    example_prompt = """
        You are a helpful, pattern-following assistant that translates
        corporate jargon into plain English. Translate the following:
        New synergies will help drive top-line growth.
        """

    # Count number of characters in the example_prompt
    character_count = len(example_prompt)
    print("Example prompt length in characters: " + str(character_count))

    # Count number of tokens in the example_prompt using GPT4 encoding
    token_count = count_tokens(example_prompt, "gpt-4")[0]
    print("Example prompt length in tokens: " + str(token_count))

    # Measure the ratio of tokens to characters:
    tokens_to_characters_ratio = token_count / character_count
    print("Ratio of tokens to characters: " + str(tokens_to_characters_ratio))

    # Estimate the cost of a hypothetical GPT4 completion the same length as the example_prompt
    completion_cost = estimate_cost(example_prompt, "gpt-4")
    print("Estimated cost of this GPT4 prompt + completion: " + str(completion_cost))

    # This example chat completion prompt is borrowed from OpenAI Cookbook:
    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    example_chat_prompt = [
        {
            "role": "system",
            "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "New synergies will help drive top-line growth.",
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Things working well together will increase revenue.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Let's talk later when we're less busy about how to do better.",
        },
        {
            "role": "user",
            "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",
        },
    ]

    # Count number of characters in the example_chat_prompt (including JSON markup)
    character_count = sum([len(str(message)) for message in example_chat_prompt])
    print("Example chat prompt length in characters: " + str(character_count))

    # Count number of tokens in the example_chat_prompt using GPT4 encoding
    token_count = count_tokens(example_chat_prompt, "gpt-4")[0]
    print("Example chat prompt length in tokens: " + str(token_count))

    # Measure the ratio of tokens to characters:
    tokens_to_characters_ratio = token_count / character_count
    print("Ratio of tokens to characters: " + str(tokens_to_characters_ratio))

    # Estimate the cost of a hypothetical GPT4 completion the same length as the example_chat_prompt
    completion_cost = estimate_cost(example_chat_prompt, "gpt-4")
    print(
        "Cost of this GPT4 chat prompt + completion both this length: "
        + str(completion_cost)
    )
