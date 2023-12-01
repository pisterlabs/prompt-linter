import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="The CSS code for a color like a blue sky at dusk:\n\nbackground-color: #",
  temperature=0,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=[";"]
)

# Prompt
# The CSS code for a color like a blue sky at dusk:

# background-color: #
# Sample response
# 3A5F9F
