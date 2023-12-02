import openai
import asyncio

if __name__ == "__main__":
    openai.api_key = ...
    asyncio.run(openai.force_init_pulls())

    # Now you can use openai with initialized sessions

    completion = openai.Completion.create(
        engine="davinci",
        prompt="Translate the following English text to French: 'Hello, how are you?'",
        max_tokens=50
    )

    print(completion)
