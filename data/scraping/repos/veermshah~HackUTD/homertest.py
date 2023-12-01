import openai
import os

openai.api_key = "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"

response = openai.Completion.create(
model="curie:ft-personal-2023-11-05-03-07-22",
prompt="How can I increase my credit to buy a home?",
max_tokens=100,
)

print(response.choices[0].text)
