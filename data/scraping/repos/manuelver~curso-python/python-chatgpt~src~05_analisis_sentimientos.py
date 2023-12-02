"""
Programa para analizar los sentimientos predominantes 
de un texto
"""
import openai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = api_key


def analizar_sentimientos(texto):
    """
    Analizar sentimientos con OpenAI GPT-3
    """

    prompt = f"Analiza los sentimientos del siguiente texto: '{texto}'. El sentimiento predominante es: "

    respuesta = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        n=1,
        temperature=0.5,
        max_tokens=100
    )

    return respuesta.choices[0].text.strip()


texto_para_analizar = input("Pega aquí el texto a analizar: ")

sentimiento = analizar_sentimientos(texto_para_analizar)

print(sentimiento)
