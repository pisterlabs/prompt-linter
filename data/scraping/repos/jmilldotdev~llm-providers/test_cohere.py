import asyncio
import os

from dotenv import load_dotenv

from llm_providers.cohere import CohereProvider


async def main():
    load_dotenv()
    provider = CohereProvider(
        connection_str=os.environ.get("COHERE_API_KEY"),
    )

    completion = await provider.complete(prompt="hey i'm a robot who")
    assert completion.prompt == "hey i'm a robot who"
    assert type(completion.completion_text) == str


if __name__ == "__main__":
    asyncio.run(main())
