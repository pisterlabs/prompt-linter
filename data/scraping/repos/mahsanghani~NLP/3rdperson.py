import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Convert this from first-person to third person (gender female):\n\nI decided to make a movie about Ada Lovelace.",
  temperature=0,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

# Prompt
# Convert this from first-person to third person (gender female):

# I decided to make a movie about Ada Lovelace.
# Sample response
# She decided to make a movie about Ada Lovelace.

