import anthropic
import asyncio
import os


async def main(max_tokens_to_sample: int = 200):
    c = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])

    response = await c.acompletion_stream(
        prompt=f"{anthropic.HUMAN_PROMPT} How many toes do dogs have?{anthropic.AI_PROMPT}",
        stop_sequences=[anthropic.HUMAN_PROMPT],
        max_tokens_to_sample=max_tokens_to_sample,
        model="claude-v1",
        stream=True,
    )
    async for data in response:
        print(data)


if __name__ == "__main__":
    asyncio.run(main())
