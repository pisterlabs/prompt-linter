"""
Programa para crear contenido 
y resumirlo con la API de OpenAI
"""

import openai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = api_key


def resumir_text(texto, tokens, temperatura, modelo="text-davinci-002"):
    """
    Resumir texto con OpenAI GPT-3
    """

    prompt = f"Resume el siguiente texto: {texto}\n\n"

    respuesta = openai.Completion.create(
        engine=modelo,
        prompt=prompt,
        n=1,
        temperature=temperatura,
        max_tokens=tokens
    )

    return respuesta.choices[0].text.strip()


# Bienvenida
print("Bienvenido a la aplicación de creación de contenido. \n Necesito que me des algunos datos.")

# Pedir datos
original = input("Pega aquí el artículo a resumir: ")
tokens = int(input("Tokens máximos: "))
temperatura = int(
    input("Del 1 al 10, ¿Cuánto quieres que sea de creativo el resumen?: ")) / 10

# Crear contenido
resumen = resumir_text(original, tokens, temperatura)


print(resumen)
