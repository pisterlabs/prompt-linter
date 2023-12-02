import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Decide whether a Tweet's sentiment is positive, neutral, or negative.\n\nTweet: \"I loved the new Batman movie!\"\nSentiment:",
  temperature=0,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.5,
  presence_penalty=0.0
)

# Prompt
# Decide whether a Tweet's sentiment is positive, neutral, or negative.

# Tweet: "I loved the new Batman movie!"
# Sentiment:
# Sample response
# Positive