from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-4-1106-preview",
  messages=[
    {"role": "system", "content": "You are a software helping people with productivty. Give tips for how to help people be more productive."},
  ]
)

print(completion.choices[0].message.content)