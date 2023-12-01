# Paquetes de OpenAI
import openai

# Paquetes de FastAPI
from fastapi import FastAPI
from fastapi import UploadFile

# Paquete de PyPDF2
from PyPDF2 import PdfReader

# Otras dependencias
from dotenv import load_dotenv
import os

"""
## 3. Entregables

En esta sección se describen los entregables relativos a FastAPI, un script en Python que 
permita crear el prototipo de un API para intercambiar información de texto con ChatGPT y generar 
ciertas consultas de intéres. Para ello se deberá crear las cuentas de plataforma y generar el 
API Key correspondiente (**NO SE DEBE INCLUIR LA API KEY**, revise 
https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety para ver como 
invocarle como variable de ambiente desde Python).


A. Diseña un API desde el framework FastAPI (denominado `fastapi_chatgpt.py`)de forma que:

    1) Deberá ser capaz de recibir peticiones tipo POST que acepten a) instrucciones en texto 
	para chaGPT y b) fragmentos de texto que ChatGPT debe analizar

    2) La acción anterior deberá retornar la respuesta de ChatGPT correspondiente a la instrucción 
	dada en el prompt en el idioma español.

**Ejemplo**
```
Indicacion: <Indicaciones del prompt empleado>. El texto que deberás analizar es el siguiente: 
<texto a analizar>

ChatGPT: Respuesta
```

Adicionalmente se deberá adjuntar capturar de pantalla en formato .png donde se aprecia el cuerpo 
de las conversaciones generadas por los ChatBots, se pueden usar numeraciones sucesivas sin son 
muchas fotos, ejemplo: evidencia_1_conversacion_i.png, evidencia_2_conversacion_i.png, ..., 
evidencia_5_conversacion_i.png

B. Se deberá adjunta un video en formato .mp4 en el que se pruebe mediante requests de Python se 
aprecie el funcionamiento del programa para obtener la respuesta que el API anterior al solicitar 
un resumen a manera de 5 bullets del contenido del archivo `news_el_economista.txt` en el idioma 
inglés.
"""

### Preparamos el API Key de OpenAI ###
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

### Preparamos FastAPI ###
app = FastAPI()

### Desarrollo del API ###

# Creamos un endpoint para saber si todo está bien
@app.post("/", status_code=200, tags=["Status"])
def all_good():
    return "All good, all working!"

# Creamos el endpoint para recibir las instrucciones y el texto a analizar
@app.post("/chatgpt", status_code=200, tags=["ChatGPT"])
def chatgpt(prompt: str, file: UploadFile):

    filename = file.filename

    file_content = file.file.read()

    # Se asegura que exista la carpeta /tmp
    # en el directorio de trabajo
    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")

    # escribe una 
    with open('./tmp/' + "copy_"+filename, "wb") as f:
        f.write(file_content)

    file_readed = detectar_tipo_archivo('./tmp/' + "copy_"+filename)

    if file_readed == None:
        return "Tipo de archivo desconocido"

    response = call_chatgpt(prompt, file_readed)

    return response


# Metodo para llamar a chatGPT mediante su API, recibe el prompt y el texto a analizar

def call_chatgpt(prompt: str, text: UploadFile):
    """
    Esta funcion es para llamar a ChatGPT mediante su API,
    recibiendo el un prompt y el archivo a analizar, retornando
    la respuesta de ChatGPT

    Parameters:
        prompt (str): El prompt para ChatGPT
        text (UploadFile): El archivo a analizar
    
    Returns:
        str: La respuesta de ChatGPT

    """
    
    chat_completion = openai.chat.completions.create(
        model = "gpt-3.5-turbo",
        temperature = 0.5,
        messages = [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": prompt + "\n'''\n" + text + "\n'''"
            }
        ]
    )
    # Imprime el mensaje respuesta de ChatGPT
    return chat_completion.choices[0].message.content


def detectar_tipo_archivo(ruta_archivo: UploadFile):

    """
    Esta funcion es para detectar el tipo de archivo que se
    recibe como parametro, retornando su contenido

    Parameters:
        ruta_archivo (UploadFile): La ruta del archivo a analizar
    
    Returns:
        str: El contenido del archivo
    """

    # Obtiene la extensión del archivo desde su ruta
    _, extension = os.path.splitext(ruta_archivo)

    # Verifica si la extensión es .txt o .pdf
    if extension == ".txt":
        with open(ruta_archivo, "r") as f:
            return f.read()
    
    elif extension == ".pdf":
        reader = PdfReader(ruta_archivo)
        page = reader.pages[0]
        return page.extract_text()
    
    else:
        return None
