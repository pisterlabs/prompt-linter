from dotenv import load_dotenv
load_dotenv()
import openai
import os

print(os.getenv("PINKY_PROMPT"))
completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": os.getenv("PINKY_PROMPT")}
  ]
)

print(completion.choices[0].message)