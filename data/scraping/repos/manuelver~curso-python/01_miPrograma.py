"""
Primer programa con chatgpt
"""

import os
import openai
import spacy
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

openai.api_key = api_key

# modelos = openai.Model.list()

# print(modelos)

modelo = "text-davinci-002"
prompt = "Cuenta una historia breve de algún lugar concreto"

respuesta = openai.Completion.create(
    engine=modelo,
    prompt=prompt,
    n=1,  # Opcional. Número de respuestas
    temperature=1,  # Opcional. Controla la creatividad de la respuesta
    max_tokens=200  # Opcional. Número máximo de tokens en la respuesta
)

# print(respuesta)

texto_generado = respuesta.choices[0].text.strip()
print(texto_generado)


# for idx, opcion in enumerate(respuesta.choices):
#     texto_generado = opcion.text.strip()
#     print(f"Respuesta {idx + 1}: {texto_generado}\n")

print("***")

modelo_spacy = spacy.load("es_core_news_md")

analisis = modelo_spacy(texto_generado)

# for token in analisis:
#     print(token.text, token.pos_, token.dep_, token.head.text)

ubicacion = None

for ent in analisis.ents:
    # print(ent.text, ent.label_)
    if ent.label_ == "LOC":
        ubicacion = ent
        break

if ubicacion:
    prompt2 = f"Dime más acerca de {ubicacion}"
    respuesta2 = openai.Completion.create(
        engine=modelo,
        prompt=prompt2,
        n=1,
        temperature=1,
        max_tokens=100
    )

    print(respuesta2.choices[0].text.strip())
