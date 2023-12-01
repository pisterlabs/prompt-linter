import os
import openai
import argparse
import re

MAX_INPUT_LENGTH = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    args = parser.parse_args()
    user_input = args.input

    print(f"User input: {user_input}")
    if not validate_length(user_input):
        raise ValueError(f"Input must be less than {MAX_INPUT_LENGTH} characters.")

    generate_branding_snippet(user_input)
    generate_keywords(user_input)


def validate_length(prompt: str):
    return len(prompt) <= MAX_INPUT_LENGTH


def generate_branding_snippet(prompt: str):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    enriched_prompt = f"Generate upbeat branding snippet for {prompt}"
    # print(enriched_prompt)
    completion = generate_chat_completion(
        "Generate upbeat branding snippet for the prompt entered by the user",
        prompt,
    )
    branding_text = completion.strip(" \n,")

    last_char = branding_text[-1]
    if last_char not in {".", "!", "?"}:
        branding_text += "..."
    print(f"Snippet: {branding_text}")

    return branding_text


def generate_keywords(prompt: str):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    completion = generate_chat_completion(
        "Generate related branding keywords for the prompt entered by the user",
        prompt,
    )
    # print(completion)
    keywords_array = re.split(",|\n|;|-", completion)
    keywords_array = [k.lower().strip().lstrip("0123456789.- ") for k in keywords_array]
    keywords_array = [k for k in keywords_array if len(k) > 0]
    print(f"Keywords: {keywords_array}")

    return keywords_array


def generate_chat_completion(system_prompt, user_prompt):
    parameters = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 500,
    }
    response = openai.ChatCompletion.create(**parameters)
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    main()
