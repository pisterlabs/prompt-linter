import openai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

def classify_text(text, model="text-davinci-002"):
  categories = ["arte", "ciencia", "deportes", "historia", "literatura", "tecnología", "viajes", "salud", "política", "religion"]
  prompt = f"Por favor clasifica el siguiente texto: '{text}' en una de las siguientes categorías: {','.join(categories)}. La categoría es: "
  response = openai.Completion.create(
    engine=model,
    prompt=prompt,
    n=1,
    max_tokens=100,
    temperature=0.5,
  )

  return response['choices'][0]['text'].strip()

text = input("Introduce un texto: ")
category = classify_text(text)

print(category)