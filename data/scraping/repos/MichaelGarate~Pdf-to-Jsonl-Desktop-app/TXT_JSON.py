import os
import re
import unicodedata
import jsonlines
import openai
from dotenv import load_dotenv

#Archivo TXT
nombre_archivo_txt = 'archivosTXT/textoPDF.txt'

#Carga las variables de entorno desde .env / cargar API de OpenAI
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
def crear_archivo_jsonl(nombre_archivo_txt, nombre_archivo_jsonl, num_preguntas=1):
    # Abre el archivo de texto y lee su contenido
    with open(nombre_archivo_txt, 'r') as f:
        texto = f.read()

    #Crear una pregunta a partir del texto utilizando GPT-3
    def generar_pregunta(texto):
        pregunta = openai.Completion.create(
            engine="text-davinci-003",
            prompt=(f"Generar una pregunta a partir del siguiente texto:\n{texto}\n\nPregunta:"),
            max_tokens=4,
            n=1,
            stop=None,
            temperature=0.6,
        )
        #convertir respuesta a minusculas y eliminar signo '¿' al inicio de la pregunta
        pregunta_generada = pregunta.choices[0].text.strip().lower()
        pregunta_generada = ''.join(c for c in unicodedata.normalize('NFD', pregunta_generada) if unicodedata.category(c) != 'Mn')
        pregunta_generada = re.sub(r'^¿', '', pregunta_generada)

        return pregunta_generada

    #Crear una respuesta a partir de las preguntas utilizando GPT-3
    def generar_respuesta(pregunta):
        respuesta = openai.Completion.create(
            engine="text-davinci-003",
            prompt=(f"Responder a la siguiente pregunta:\n{pregunta}\n\nRespuesta:"),
            max_tokens=2,
            n=1,
            stop=None,
            temperature=0.6,
        )

        #Convertir respuesta a minusculas
        respuesta_generada = respuesta.choices[0].text.strip().lower()
        respuesta_generada = ''.join(c for c in unicodedata.normalize('NFD', respuesta_generada) if unicodedata.category(c) != 'Mn')

        return respuesta_generada

    # Verifica si el archivo JSONL ya existe y lee las preguntas existentes
    preguntas_existentes = set()
    if os.path.isfile(nombre_archivo_jsonl):
        with jsonlines.open(nombre_archivo_jsonl, mode='r') as reader:
            for obj in reader:
                pregunta = obj['prompt']
                preguntas_existentes.add(pregunta)

    #Crear una lista de preguntas a partir del texto utilizando OpenAI GPT-3
    preguntas = []
    preguntas_generadas = set()  #Almacena las preguntas generadas (sin repetir)
    while len(preguntas) < num_preguntas:
        pregunta_generada = generar_pregunta(texto)
        # Si la pregunta creada no esta en preguntas_generadas y no existe en el archivo JSONL, la agregamos a preguntas
        if pregunta_generada not in preguntas_generadas and pregunta_generada not in preguntas_existentes:
            preguntas_generadas.add(pregunta_generada)
            preguntas.append(pregunta_generada)

    #
    # Crea una respuesta para cada pregunta utilizando OpenAI GPT-3
    respuestas = []
    for pregunta in preguntas:
        respuesta_generada = generar_respuesta(pregunta)
        # Verifica si la pregunta ya existe en el archivo JSONL antes de agregarla a respuestas
        if pregunta not in preguntas_existentes:
            respuestas.append(respuesta_generada)
        else:
            print(f"La pregunta '{pregunta}' ya existe en el archivo JSONL y no se agregará una respuesta para ella")

    # Verifica si el archivo JSONL ya existe
    if os.path.isfile(nombre_archivo_jsonl):
        # Si el archivo existe, abre el archivo en modo "a" para agregar las nuevas preguntas y respuestas al final del archivo
        with jsonlines.open(nombre_archivo_jsonl, mode='a') as writer:
            for pregunta, respuesta in zip(preguntas, respuestas):
                if pregunta is not None and respuesta is not None:
                    writer.write({"prompt": pregunta, "completion": respuesta})
    else:
        # Si el archivo no existe, abre el archivo en modo "w" para crear un nuevo archivo y escribir las preguntas y respuestas
        with jsonlines.open(nombre_archivo_jsonl, mode='w') as writer:
            for pregunta, respuesta in zip(preguntas, respuestas):
                if pregunta is not None and respuesta is not None:
                    writer.write({"prompt": pregunta, "completion": respuesta})
