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


def traducir_texto(texto, idioma):
    """
    Analizar sentimientos con OpenAI GPT-3
    """

    prompt = f"Traduce el siguiente texto al idioma {idioma}:\n\n{texto}\n\nTexto traducido: "

    respuesta = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        n=1,
        temperature=0.5,
        max_tokens=1000
    )

    return respuesta.choices[0].text.strip()


print("Bienvenido al traductor de texto\n")

idioma = input("Escribe el idioma al que quieres traducir: ")
texto_a_traducir = input("Escribe el texto a traducir: ")

texto_traducido = traducir_texto(texto_a_traducir, idioma)

print(f"El texto traducido es: {texto_traducido}")
