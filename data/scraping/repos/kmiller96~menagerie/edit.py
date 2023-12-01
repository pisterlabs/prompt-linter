"""Edits the article content to improve its grammar and spelling."""

import os
import sys
import openai

## Load the article ##
with open(sys.argv[-1], "r", encoding="utf-8") as f:
    article = f.read()

## Submit to ChatGPT ##
openai.api_key = os.getenv("OPENAI_SECRET")

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": (
                "You are an editor for a blog. You have been given a blog "
                "article which you need to review for spelling and grammar "
                "issues. Take the following article with any spelling and "
                "grammar issues fixed. Do not respond with anything except the "
                "corrected article."
            ),
        },
        {
            "role": "user",
            "content": article,
        },
    ],
)

## Print the response ##
print(completion["choices"][0]["message"]["content"])
