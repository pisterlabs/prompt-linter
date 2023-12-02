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


prompt = "2+2="

response = chat(prompt)
print(response)

prompt = """When I was 12, my sister was half my age. When I turn 70, what will her age be?"""

response = chat(prompt)
print(response)
