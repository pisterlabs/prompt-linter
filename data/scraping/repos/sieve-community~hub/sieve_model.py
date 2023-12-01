import sieve

@sieve.function(
    name="openai-gpt",
    python_version="3.8",
    system_packages=["libgl1-mesa-glx", "libglib2.0-0"],
    python_packages=[
        "openai==0.11.3",
    ],
    environment_variables=[
        sieve.Env(name="OPENAI_API_KEY", description="Your OpenAI API key"),
        sieve.Env(name="OPENAI_MODEL", description="OpenAI completions model", default="text-davinci-003"),
        sieve.Env(name="temperature", description="Temperature value. Controls the randomness of the generated result.", default=0.9),
        sieve.Env(name="frequency_penalty", description="Frequency penalty that adjusts tokens based on their frequency. Positive value discourages frequent tokens, while negative value encourages them.", default=0),
        sieve.Env(name="presence_penalty", description="Presence penalty that rewards or penalizes tokens based on their presence in the prompt. A positive value penalizes repeated usage, while a negative value encourages it.", default=0.6),
        sieve.Env(name="max_tokens", description="The maximum number of tokens to generate in the completion.", default=100)
    ]
)
def openai_completion(prompt: str) -> str:
    import openai
    import os
    openai.api_key = os.environ["OPENAI_API_KEY"]

    response = openai.Completion.create(
        engine=os.environ["OPENAI_MODEL"],
        prompt=prompt,
        max_tokens=int(os.environ["max_tokens"]),
        temperature=float(os.environ["temperature"]),
        top_p=1,
        frequency_penalty=float(os.environ["frequency_penalty"]),
        presence_penalty=float(os.environ["presence_penalty"]),
    )
    return response.choices[0].text

@sieve.workflow(name="test_openai-gpt")
def test_openai_completion(prompt: str) -> str:
    return openai_completion(prompt)