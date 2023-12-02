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


def clasificar_texto(texto):
    """
    Clasificar texto con OpenAI GPT-3
    """

    # Definir categorías en un array
    categorias = [
        "Arte",
        "ciencia",
        "deportes",
        "entretenimiento",
        "educación",
        "finanzas",
        "historia",
        "literatura",
        "matemáticas",
        "medicina",
        "medio ambiente",
        "música",
        "noticias",
        "política",
        "religión",
        "salud",
        "tecnología",
        "viajes",
    ]

    prompt = f"Clasifica el siguiente texto: '{texto}' en una de estar categorías: {','.join(categorias)}. La categoría es: "

    respuesta = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        n=1,
        temperature=0.5,
        max_tokens=50
    )

    return respuesta.choices[0].text.strip()


texto_para_clasificar = input("Ingresa texto a clasificar en una categoría: ")

clasificacion = clasificar_texto(texto_para_clasificar)

print(clasificacion)
