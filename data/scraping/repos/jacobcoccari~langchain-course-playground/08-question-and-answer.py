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


prompt = """What are the major sub-segments of psychology?"""
response = chat(prompt)
print(response)

prompt = """What are the major sub-segments of psychology and can you please also define these types of psychology and provide this in a markdown table?"""
response = chat(prompt)
print(response)
