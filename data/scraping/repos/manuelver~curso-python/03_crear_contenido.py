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


def crear_contenido(tema, tokens, temperatura, modelo="text-davinci-002"):
    """
    Crea contenido con OpenAI GPT-3
    """

    prompt = f"Escribe un artículo corto sobre el tema: {tema}"

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
tema = input("Elige un tema para tu artículo: ")
tokens = int(input("Tokens máximos: "))
temperatura = int(
    input("Del 1 al 10, ¿Cuánto quieres que sea de creativo el artículo?: ")) / 10

# Crear contenido
articulo_creado = crear_contenido(tema, tokens, temperatura)


print(articulo_creado)
