import os
import openai

# Configuration of the API key
with open("fine_tuning/key.txt", "r") as file:
    api_key = file.read().strip()
    
openai.api_key = api_key

response = openai.Completion.create(
  model="ft:babbage-002:personal::8FPOuwnM",
  prompt="In the following tweet what is the rate of sadness over 10? here is the tweet: I feel like singing the song human sadness by Julian Cassablancas and The Voidz or some other sad music. #depression #sad #sadness ->",
  temperature=0.2,
  max_tokens=10,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=["\n"]
)

print(response)