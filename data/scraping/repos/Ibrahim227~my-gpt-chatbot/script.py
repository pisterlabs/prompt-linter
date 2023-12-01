import os
from dotenv import load_dotenv
import openai


os.environ["OPENAI_API_KEY"] = "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"

print(os.environ.get("OPENAI_API_KEY"))

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# connect to an API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Example OpenAI Python library request
MODEL = "gpt-3.5-turbo"
response = openai.ChatCompletion.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Knock knock."},
        {"role": "assistant", "content": "Who's there?"},
        {"role": "user", "content": "Orange."},
    ],
    temperature=0,
)

response()
