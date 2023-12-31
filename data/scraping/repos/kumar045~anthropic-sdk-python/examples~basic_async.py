import asyncio
import os
import anthropic


async def main(max_tokens_to_sample: int = 100):
    c = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])
    resp = await c.acompletion(
        prompt=f"{anthropic.HUMAN_PROMPT} How many toes do dogs have?{anthropic.AI_PROMPT}",
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1",
        max_tokens_to_sample=max_tokens_to_sample,
    )
    print(resp)


if __name__ == "__main__":
    asyncio.run(main())
