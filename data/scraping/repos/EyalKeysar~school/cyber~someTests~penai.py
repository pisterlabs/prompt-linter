import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is python"

response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    temperature=0.4,
    max_tokens=100,
)
print(response)
