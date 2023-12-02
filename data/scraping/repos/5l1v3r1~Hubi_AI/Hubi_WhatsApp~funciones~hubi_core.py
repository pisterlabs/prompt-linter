# Core de Hubi.
# by @flowese
# powered by @hubspain

#Librerías.
from dotenv import load_dotenv
import os
from os import getenv
from os import system
from random import choice
import openai
from flask import Flask, request
import datetime
import segno
import json

print ('· ESTADO CORE HUBI: OK')
# Interacciones con GPT-3 API por defecto.
interaccion = 0
# Memoria de Hubi por defecto.
memoria_hubi = 'hubi_mem'

# Cargamos dotenv y variables del fichero .env.
load_dotenv()

# API OpenAI
openai.api_key = getenv("OPENAI_API_KEY")
gpt3apikey = openai.api_key
print(f'· OPENAI API KEY: {gpt3apikey}')
completion = openai.Completion()
# Path de memoria de Hubi.
mempath = getenv("MEM_PATH")
print (f'· PATH MEMORIA: {mempath}')
# Dominio configurado.
dominio_hubi = getenv("DOMAIN")
# Check por consola
print (f'· DOMINIO HOST: {dominio_hubi}')
# Tipo de memoria Custom.
customem = getenv("CUSTOM_MEM")
# Check por consola
print (f'· CUSTOM MEM: {customem}')

# Path de logs.
logpath = getenv("LOG_PATH")
# Check por consola
print (f'· PATH LOG: {logpath}')

# Clave secreata from env.
clave_secreta = getenv("SECRET_KEY")
# Check por consola
print (f'· UNIKEY HUBI: {clave_secreta}')


# funcion de inicio que crea log si no existe y guardamos el anterior si existe.
def inicio_app():
    # Variables de Twilio.
    twil_Num = getenv("TWILIO_NUM")
    twil_JoinID = getenv("TWILIO_SANDBOX_WORD")
    # Check por consola
    print (f'· TWILIO WHATSSAPP NUM.: {twil_Num}\n· TWILIO JOIN WORD: {twil_JoinID}')
    comprobarlog = os.path.isfile(logpath)
    now = datetime.datetime.now()
    fechalog = (now.strftime("%Y%m%d_%H%M%S"))
    if comprobarlog is False:
        nombre_nuevo = f'{logpath}/archivo/HUBI_{fechalog}.log'
        os.rename(f'{logpath}/chatlog.log', nombre_nuevo)
    logfile = open (f'{logpath}/chatlog.log','a')
    logfile.write(str(f'\n{fechalog} - ¡Hubi ha despertado!\n'))
    logfile.close()
    # Parametros de arranque de la App + creamos QR para WhatsApp Twilio.
    # Generando QR con los datos anteriores.
    qr = segno.make(f'whatsapp://send?phone={twil_Num}&text={twil_JoinID}')
    # Guardamos imgaen de qr a escala 5 de tamaño.
    qr.save('Hubi_WhatsApp/static/qr_whatsapp.png', scale=5)

# Función que carga variables de sistema.
def variables_sistema():
    # Carga de variables de entorno.
    now = datetime.datetime.now()
    fecha_now = (now.strftime("%d-%m-%Y"))
    hora_now = (now.strftime("%H:%M:%S"))
    system('clear')
    # inicio por consola cross-check.
    print (f'\n· HUBI VERSION: 1.5(dev)\n· FECHA ACTUAL: {fecha_now}\n· HORA ACTUAL: {hora_now}')
    clave_secreta = getenv("SECRET_KEY")
    logpath = getenv("LOG_PATH")
    debug_server = getenv("FLASK_DEBUG")
    server_port = getenv("FLASK_RUN_PORT")
    server_ip = getenv("FLASK_RUN_HOST")
    return fecha_now, hora_now, clave_secreta, server_port, server_ip, debug_server

# Función que lee archivos json 
def importarjson(fichero):
    with open(fichero, 'r') as fichero:
        fichero = json.load(fichero)
        return fichero

# Secuencias de inicio y reinicio para GPT-3.
nombre_persona = 'Persona'
start_sequence = "\nHubi:"
restart_sequence = f"\n\n{nombre_persona}:"

# Sentencia a GPT-3 con los parámetros.
def sentencia(question, chat_log=None):
    prompt_text = f'{chat_log}{restart_sequence}: {question}{start_sequence}:'
    response = openai.Completion.create(
      engine="davinci",
      prompt=prompt_text,
      temperature=0.7,
      max_tokens=90,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0.3,
      stop=["\n"],
    )
    story = response['choices'][0]['text']
    return str(story)

# Sentencia a GPT-3 Custom.
def sentencia_custom(motivo):
    prompt_text = f'{motivo}:'
    response = openai.Completion.create(
      engine="davinci",
      prompt=prompt_text,
      temperature=0.7,
      max_tokens=90,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0.3,
      stop=["\n"],
    )
    story = response['choices'][0]['text']
    return str(story)

# Directorio de ayuda.
def help():
    ayuda = open('Hubi/static/ayuda.help','r')
    content = ayuda.read()
    return content
                   
# Insertar metadatos fecha en log.
def insert_logmeta(interaccion,memoria_activa):
    now = datetime.datetime.now()
    fechalog = (now.strftime("%Y-%m-%d %H:%M:%S"))
    logfile = open (f'{logpath}/chatlog.log','a')
    logfile.write(str(f'\n\n{fechalog} - Interacción: {interaccion} \n{fechalog} - Memoria Activa: {memoria_activa}\n'))
    logfile.close()

# Reset memoria
def reset_memoria(tipo_memoria):
    global interaccion
    global memoria_hubi
    global mempath
    global customem
    if tipo_memoria == 'hubi_mem' or tipo_memoria == 'apper_mem' or tipo_memoria == 'avoris_mem' or tipo_memoria == 'hubspain_mem' or tipo_memoria == customem:
        memoria_hubi = f'/{tipo_memoria}'
        return f'La memoria de Hubi ha sido reiniciada a {tipo_memoria}.'
    if tipo_memoria == 'corta':
        interaccion = 20
        desc_mem = 'Hubi ahora es más basico y es más aleatorio.'
        return f'La memoria de Hubi ha sido reiniciada a memoria {tipo_memoria}.\n{desc_mem}'
    if tipo_memoria == 'media':
        interaccion = 7
        desc_mem = 'La inteligencia de Hubi ahora no es tan buena.'
        return f'La memoria de Hubi ha sido reiniciada a memoria {tipo_memoria}.\n{desc_mem}'
    if tipo_memoria == 'larga':
        interaccion = 0
        desc_mem = 'La memoria larga hace que sea muy inteligente y más preciso.'
        return f'La memoria de Hubi ha sido reiniciada a memoria {tipo_memoria}.\n{desc_mem}'
    else:
        return f'El tipo {tipo_memoria} no existe.'

# Gestión de memoria de GPT-3.
def memoria_interaccion(question, answer, chat_log=None):
    global interaccion
    global memoria_hubi
    interaccion+=1
    f1 = open (f'{mempath}/{memoria_hubi}/memoria_larga.data','r')
    mem_largo_plazo = f1.read()
    f2 = open (f'{mempath}/{memoria_hubi}/memoria_media.data','r')
    mem_medio_plazo = f2.read()
    f3 = open (f'{mempath}/{memoria_hubi}/memoria_corta.data','r')
    mem_corto_plazo = f3.read()
    if interaccion < 5 or None:
        chat_log = mem_largo_plazo
        memoria_activa = 'Largo Plazo'
        # Insertar datos en log.
        insert_logmeta(interaccion,memoria_activa)
        return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'
    if interaccion < 10:
        chat_log = mem_medio_plazo
        memoria_activa = 'Medio Plazo'
        # Insertar datos en log.
        insert_logmeta(interaccion,memoria_activa)
        return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'
    else:
        chat_log = mem_corto_plazo
        memoria_activa = 'Corto Plazo'
        # Insertar datos en log.
        insert_logmeta(interaccion,memoria_activa)
        return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'