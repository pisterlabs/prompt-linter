import os
import openai
openai.api_type = "azure"
openai.api_base = "https://da-stg-openai-eu-fr-m6allrbs2iz3e.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = 'a3ef17c3d1ac4a8b85e899994172b12a' # os.getenv("OPENAI_API_KEY")

file = open("./snake3.rb", "r")

response = openai.ChatCompletion.create(
  engine="gpt-4-32k",
  messages = [
    { "role": "user", "content": 'Convert this ruby source code to C#.' },
    { "role": "user", "content": 'Remain variable and class names unchanged as much as possible.' },
    { "role": "user", "content": file.read() }
  ],
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)

print(response)
