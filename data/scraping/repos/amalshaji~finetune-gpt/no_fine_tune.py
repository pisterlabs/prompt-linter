import asyncio
import timeit
import openai
import os
from string import Template

openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = Template(
    """
Generate a message in less than 50 words using the following parameters:
    occasion: $occasion
    tone: $tone

Use the following template tags as placeholders wherever necessary. You can use \
them multiple times. Do not use unknown tags or placeholders.
| tags | description |
|---|---|
| {{ first_name }} | User's first name |
| {{ last_name }} | User's last name |
| {{ email }} | User's email |
"""
)


async def run(occasion: str, tone: str):
    start = timeit.default_timer()
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt.substitute({"occasion": occasion, "tone": tone}),
            }
        ],
    )
    print(response)
    print(f"Time taken: {timeit.default_timer() - start}s")


asyncio.run(run(occasion="birthday", tone="texas accent"))
