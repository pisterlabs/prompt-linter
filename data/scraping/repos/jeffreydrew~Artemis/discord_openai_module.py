import os
import openai

openai.organization = os.environ.get("OPENAI_ORG")
openai.api_key = os.environ.get("OPENAI_TOKEN")


def gpt_query(prompt = None, model = "gpt-3.5-turbo-0301", temperature = 0.5, max_tokens = 2000):
    if prompt is None:
        return "No prompt detected, please ask again"
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

print(gpt_query("Are you ready?"))
