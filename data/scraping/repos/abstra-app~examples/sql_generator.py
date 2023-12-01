from abstra.forms import *
import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

ans = (
    Page()
    .display_markdown(
        """
## Welcome to your fav SQL query generator
"""
    )
    .read("Enter your query prompt below:", key="prompt")
    .run()
)

response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Give me a sql query example for the expecified word: " + ans["prompt"],
    temperature=0.3,
    max_tokens=100,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)

query = response["choices"][0]["text"]

display(query)
