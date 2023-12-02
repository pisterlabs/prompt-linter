import environ
import openai
from .models import *

env = environ.Env()
environ.Env.read_env()
openai.api_key = env("OPENAI_API_KEY")
print("\n\nAPI KEY CONFIGURADA: ", env('OPENAI_API_KEY'),"\n\n")

system_message = 'Eres un sistema de apoyo para astrónomos en la búsqueda de archivos astronómicos en el IVOA, debes chatear con el astrónomo y averiguar lo que necesita para luego entregar la consulta que deberá hacer con la librería PyVO de python y resolver sus dudas al respecto, el usuario conoce y tiene instalada la libreria de PyVO y Astropy por lo que solo quiere la consulta que debe realizar a partir de algún dato del Registry, cuando el usuario indique el comando "QUERY" entregarás la consulta para hacer en PyVO que obtuviste al chatear con el astrónomo'

def processChat(chat_anterior):
    messages = [
        {"role": "system", "content":system_message},
    ]
    for mensaje in chat_anterior:
        if(mensaje.usuario == "USER"):
            messages.append({"role":"user", "content":mensaje.mensaje})
        elif(mensaje.usuario =='GPT'):
            messages.append({"role":"assistant", "content":mensaje.mensaje})
    return messages

def makeGPTquery(chat,u_message):
    chat_anterior = Mensajes.objects.filter(id_chat=chat)
    if(len(chat_anterior) > 0):
        messages = processChat(chat_anterior)
    else:
        messages=[
            {"role": "system", "content":system_message},
            {"role":"user", "content":u_message},  
        ]

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        temperature = 1,
        messages= messages
    )
    nMessage = Mensajes()
    nMessage.id_chat = chat
    nMessage.usuario = 'GPT'
    nMessage.mensaje = response['choices'][0]['message']['content']
    nMessage.save()

    return nMessage