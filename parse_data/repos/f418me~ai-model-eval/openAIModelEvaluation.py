import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("No API key found. Please add it to the .env file.")

os.environ["OPENAI_API_KEY"] = api_key


# Create an OpenAI client
client = OpenAI()

# Models that should be tested
models = ["gpt-4-1106-preview", "gpt-4", "gpt-3.5-turbo-1106","gpt-3.5-turbo"]

# Different temperatures that should be tested to see how they affect the answers
temperatures = [0, 0.5, 1, 1.5]

# Messages that should be used as context
messages = [
    {"role": "system", "content": "You are Satoshi Nakamoto."},
    {"role": "user", "content": "Why did you decide to remain anonymous?"},
]

# Öffne eine Datei zum Schreiben der Antworten
with open('openai_model_responses.txt', 'w') as file:
    for model in models:
        for temp in temperatures:
            response = client.chat.completions.create(
                model=model,
                temperature=temp,
                max_tokens=1024,
                messages=messages
            )

            # Schreibe den Modellnamen, die Temperatur und die Antwort in die Datei
            file.write(f'model: {model}\n')
            file.write(f'temperature: {temp}\n')
            file.write(f'answer: {response.choices[0].message.content}\n\n')

print("Done, the answers have been saved.")
