import os
import openai

openai.api_key = ""

response = openai.Completion.create(
  prompt="Де знаходиться музей історії Києва ? \nA:",
  model="text-davinci-003",
  temperature=0,
  max_tokens=100,
  top_p=1,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["\n"]
)
print (response['choices'][0]['text'])