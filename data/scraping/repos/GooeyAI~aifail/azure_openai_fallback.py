import os

import openai

from aifail import retry_if, try_all, openai_should_retry

azure_client = openai.AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version="2023-10-01-preview",
    max_retries=0,  # disables openai's internal retry logic
)
openai_client = openai.OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    max_retries=0,  # disables openai's internal retry logic
)


@retry_if(openai_should_retry)
def chad_gpt4(messages, max_tokens=4000):
    response = try_all(
        # first try with azure
        lambda: azure_client.chat.completions.create(
            model="openai-gpt-4-prod-ca-1",  # replace with your deployment name
            messages=messages,
            max_tokens=max_tokens,
        ),
        # if that fails (for whatever reason), try with openai
        lambda: openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=max_tokens,
        ),
    )
    print(response.choices[0].message.content)


chad_gpt4(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant designed to output JSON.",
        },
        {
            "role": "user",
            "content": "Who won the world series in 2020?",
        },
    ]
)
