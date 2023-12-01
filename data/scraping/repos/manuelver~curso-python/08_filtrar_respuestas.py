"""
Chatbot con OpenAI GPT-3
"""

import openai
import os
import spacy
import numpy as np
from dotenv import load_dotenv
from colorama import init, Fore

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = api_key

preguntas_anteriores = []
respuestas_anteriores = []
modelo_spacy = spacy.load("es_core_news_md")
palabras_prohibidas = ["palabra1", "palabra2", "paella"]


# Inicializar colorama
init()


def clearConsole():
    """
    Limpia la consola
    """

    os.system('clear')


def similitud_coseno(vec1, vec2):
    """
    Calcula la similitud coseno entre dos vectores
    """

    superposicion = np.dot(vec1, vec2)
    magnitud1 = np.linalg.norm(vec1)  # Longitud del vector
    magnitud2 = np.linalg.norm(vec2)  # Longitud del vector

    sim_cos = superposicion / (magnitud1 * magnitud2)

    return sim_cos


def es_relevante(respuesta, entrada, umbral=0.2):
    """
    Determina si una respuesta es relevante para una entrada
    """

    entrada_vectorizada = modelo_spacy(entrada).vector
    respuesta_vectorizada = modelo_spacy(respuesta).vector

    # Ahora que tenemos las anteriores variables transformadas en vectores,
    # podemos calcular la similitud coseno
    similitud = similitud_coseno(entrada_vectorizada, respuesta_vectorizada)

    # Si la similitud es mayor o igual al umbral, la respuesta es relevante
    return similitud >= umbral


def filtrar_lista_negra(texto, lista_negra):
    """
    Filtra palabras de una lista negra
    """

    # Separar el texto en tokens
    token = modelo_spacy(texto)
    # Crear una lista vacía para guardar los tokens permitidos
    resultado = []

    for t in token:
        # Si el token no está en la lista negra, agregarlo al resultado
        if t.text not in lista_negra:
            resultado.append(t.text)

        else:
            # Si el token está en la lista negra, agregarlo al resultado
            resultado.append("[xxxxx]")

    return " ".join(resultado)


def preguntar_chat_gpt(prompt, modelo="text-davinci-002"):
    """
    Pregunta a la API de OpenAI GPT-3
    """

    respuesta = openai.Completion.create(
        engine=modelo,
        prompt=prompt,
        n=1,
        temperature=1,
        max_tokens=150
    )

    respuesta_sin_filtrar = respuesta.choices[0].text.strip()

    respuesta_filtrada = filtrar_lista_negra(
        respuesta_sin_filtrar, palabras_prohibidas)

    return respuesta_filtrada


# Bienvenida
clearConsole()
print(Fore.BLUE + "Bienvenido al chatbot de OpenAI GPT-3." + Fore.RESET)
print(Fore.BLUE + "Escribe \"salir\" cuando quieras terminar la conversación." + Fore.RESET)

# Loop para controlar el flujo de la conversación
while True:

    conversacion_historica = ""

    ingreso_usuario = input(Fore.MAGENTA + "Tú: " + Fore.RESET)

    if ingreso_usuario == "salir":
        break

    for pregunta, respuesta in zip(preguntas_anteriores, respuestas_anteriores):
        conversacion_historica += f"{Fore.BLUE}Usuario pregunta: {Fore.RESET}{pregunta}"
        conversacion_historica += f"{Fore.GREEN}Bot responde: {Fore.RESET}{respuesta}\n"

    prompt = f"{Fore.CYAN}Usuario pregunta: {Fore.RESET}{ingreso_usuario}"
    conversacion_historica += prompt
    respuesta_gpt = preguntar_chat_gpt(conversacion_historica)

    relevante = es_relevante(respuesta_gpt, ingreso_usuario)

    if relevante:
        print(f"{respuesta_gpt}")

        preguntas_anteriores.append(ingreso_usuario)
        respuestas_anteriores.append(respuesta_gpt)
    else:
        print(Fore.RED + "La respuesta no es relevante ¿podrías reformularla?" + Fore.RESET)
