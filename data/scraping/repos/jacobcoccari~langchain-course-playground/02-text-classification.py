from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def chat(message, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": message}],
    )
    return response.choices[0].message["content"]


prompt = """Here are a series of customer reviews. Please classify all of them:
1. 'I am so impressed with this company'
2. 'The shipping was extremely slow'
3. 'The pricing is too high for what you get'"""

response = chat(prompt)
print(response)
