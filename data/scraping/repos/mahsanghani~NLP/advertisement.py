import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Write a creative ad for the following product to run on Facebook aimed at parents:\n\nProduct: Learning Room is a virtual environment to help students from kindergarten to high school excel in school.",
  temperature=0.5,
  max_tokens=100,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

# Prompt
# Write a creative ad for the following product to run on Facebook aimed at parents:

# Product: Learning Room is a virtual environment to help students from kindergarten to high school excel in school.
# Sample response
# Are you looking for a way to give your child a head start in school? Look no further than Learning Room! Our virtual environment is designed to help students from kindergarten to high school excel in their studies. Our unique platform offers personalized learning plans, interactive activities, and real-time feedback to ensure your child is getting the most out of their education. Give your child the best chance to succeed in school with Learning Room!