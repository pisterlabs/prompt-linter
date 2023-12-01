import openai


def generate_completion(api_key: str, model: str, prompt: str, temperature: float) -> str:
    openai.api_key = api_key

    return openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=1000,
        temperature=temperature
    )["choices"][0]["text"].strip() + "\n"
