import openai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

def analytic_feel(text, model="text-davinci-002"):
  prompt = f"Por favor, analiza el sentimiento predominante en el siguiente texto: '{text}'. El sentimiento es: "
  response = openai.Completion.create(
    engine=model,
    prompt=prompt,
    n=1,
    max_tokens=100,
    temperature=0.5,
  )

  return response['choices'][0]['text'].strip()

text = input("Introduce un texto: ")
feel = analytic_feel(text)

print(feel)