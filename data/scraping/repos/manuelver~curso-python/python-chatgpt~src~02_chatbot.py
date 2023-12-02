"""
Chatbot con OpenAI GPT-3
"""

import openai
import os
from dotenv import load_dotenv
from colorama import init, Fore

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = api_key

preguntas_anteriores = []
respuestas_anteriores = []

# Inicializar colorama
init()


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

    return respuesta.choices[0].text.strip()


# Bienvenida
print(Fore.RED + "Bienvenido al chatbot de OpenAI GPT-3." + Fore.RESET)
print(Fore.RED + "Escribe \"salir\" cuando quieras terminar la conversación." + Fore.RESET)

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

    print(f"{respuesta_gpt}")

    preguntas_anteriores.append(ingreso_usuario)
    respuestas_anteriores.append(respuesta_gpt)
