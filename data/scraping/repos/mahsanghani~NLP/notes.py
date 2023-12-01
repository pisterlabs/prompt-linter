import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="What are 5 key points I should know when studying Ancient Rome?",
  temperature=0.3,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

# Prompt
# What are 5 key points I should know when studying Ancient Rome?
# Sample response
# 1. Understand the Roman Republic and its political and social structures.
# 2. Learn about the major events and people of the Roman Empire, including the Pax Romana.
# 3. Familiarize yourself with Roman culture and society, including language, art, architecture, literature, law, and religion.
# 4. Study the Roman military, its tactics and organization, and its effects on the empire.
# 5. Examine the decline of the Roman Empire, its eventual fall, and its legacy.