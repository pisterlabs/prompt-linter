"""OpenAI utils."""
import time
import openai

# This obviously should depend on the tokenizer used but we don't have access
# to that, so we just use a heuristic. Note that this does not mean the context
# window of the model, but the context given to the agent in the context section
# of the prompt.
OPENAI_MODELS = {
    "gpt-3.5-turbo": {
        "max_length": 5000,
    },
    "gpt-3.5-turbo-16k": {
        "max_length": 5000 * 10 * 2,
    },
    "gpt-4": {
        "max_length": 5000 * 10,
    },
}

SUPPORTED_LANGUAGE_MODELS = OPENAI_MODELS


def get_completion(
    prompt, model="gpt-3.5-turbo", temperature=0, max_tokens=4000, stop=["```"]
):
    supported_models = list(SUPPORTED_LANGUAGE_MODELS.keys())
    assert (
        model in supported_models
    ), f"Model {model} not supported. Supported models: {supported_models}"

    if model in OPENAI_MODELS:
        try:
            return openai_call(
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
        except openai.error.InvalidRequestError as exc:
            return {"error": str(exc)}


def openai_call(
    prompt, model="gpt-3.5-turbo", temperature=0, max_tokens=1024, stop=["```"]
):
    """Wrapper over OpenAI's completion API."""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            temperature=temperature,
            stop=stop,
        )
        text = response["choices"][0]["message"]["content"]
    except (
        openai.error.RateLimitError,
        openai.error.APIError,
        openai.error.Timeout,
        openai.error.APIConnectionError,
    ) as exc:
        print(exc)
        print("Error from OpenAI's API. Sleeping for a few seconds.")
        time.sleep(5)
        text = openai_call( 
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

    return text
