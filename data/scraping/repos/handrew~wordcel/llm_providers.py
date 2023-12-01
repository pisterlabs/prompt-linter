import time
import openai


def openai_call(
    prompt, model="gpt-3.5-turbo", temperature=0, max_tokens=1024, stop=["```"], sleep=60
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
        openai.error.ServiceUnavailableError,
    ) as exc:
        print(exc)
        print("Error from OpenAI's API. Sleeping for a few seconds.")
        time.sleep(sleep)
        text = openai_call(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

    return text
