# FUNCIONES PRINCIPALES
# ############################################################################

# 0 - IMPORTS&SETTINGS
# 1 - Text Handling
# 2 - OpenAI embeddings & queries
# 3 - Orquestador

##################
# Imports&Settings
##################
import numpy as np
import pandas as pd
import csv
import os
import re
import io
import json

import pypdf
import tabula

import openai
import pyllamacpp
import tiktoken

import langchain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# API KEY:
# ToDo: Cambiar a variable de entorno
try:
    api_key = json.load(open('./data/creds/gpt_id.json'))['api_key']
    openai.api_key = api_key
    os.environ['OPENAI_API_KEY'] = api_key
except:
    None

###############
# Text Handling
###############
# ToDo: Hablar con Ana y Natalia sobre tratamiento de PDFs y OCR.
'''
Funciones para manejo de texto.
Principalmente extractar texto de pdfs y limpiarlo.
En un futuro se podrían anyadir funciones para extraer texto de otros formatos
e incluso de imágenes utilizando las últimas habilidades de GPT (se podrían
convertir gráficos a su equivalente en tablas de datos, json, etc.).
'''
def limpieza_texto(texto: str) -> str:
    '''
    Función para limpiar texto de pdfs.
    Cambia saltos de línea, espacios en blanco y caracteres especiales.
    '''
    # Eliminamos espacios en blanco
    texto = re.sub(' +', ' ', texto)
    # Eliminamos caracteres especiales [REVISAR]
    texto = re.sub('[^A-Za-z0-9]+', ' ', texto)
    # Eliminamos saltos múltiples de línea
    texto = re.sub(r"\n\s*\n", "\n\n", texto)
    return texto

def tabla_a_texto(tabla, anyo_actual=None):
    '''
    Función para convertir una tabla de pandas en un texto.
    La idea es identificar los nombres de columna e índices correctos y
    a partir de ahí generar un texto que pueda ser procesado por el modelo.
    '''
    tabla = tabla.copy()
    
    # Tamanyo mínimo de tabla para que sea válida = 2x2
    if sum(tabla.shape) < 4:
        return ''
    
    # Lista de valores que consideramos NaN:
    nan_equiv = [np.NaN, np.nan,
                       'nan', 'NaN', 'Nan', 'NAN', 'na', 'NA',
                       'Unnamed:0', 'Unnamed: 0'
                       '', '-', ' ', '  ', '   ']
    
    # Asumimos que el primer elemento es el título salvo si es NaN:
    titulo = tabla.columns[0] if tabla.columns[0] not in nan_equiv else ''
    
    # Asumimos que la primera columna es el índice y la eliminamos:
    tabla.index = tabla[tabla.columns[0]].values
    tabla.drop(columns=tabla.columns[0], inplace=True)

    # Si las columnas tienen muchos 'Unnamed' suele ser porque hay
    # varias líneas de texto. En ese caso, las juntamos en una sola:
    if sum(['Unnamed' in i for i in tabla.columns]) > 2:
        nueva_columna = [f'{tabla.columns[i]} {tabla.iloc[0,i]}'
                         for i in range(len(tabla.columns))]
        nueva_columna = [i.replace('Unnamed: ','') for i in nueva_columna]
        tabla.columns = nueva_columna

    
    # Eliminamos las filas y columnas que no tienen datos:
    '''
    ToDo: filas vacías con NaN muchas veces implican
    que el índice de la fila es parte del texto del índice
    de la siguiente fila. Hacer algo para que se unan
    '''
    tabla.replace(nan_equiv, np.nan, inplace=True)
    tabla.dropna(axis=0, how='all', inplace=True)
    tabla.dropna(axis=1, how='all', inplace=True)
    
    # Check si las columnas son anyos:
    col_anyos = False
    years_txt=[str(i) for i in range(2015,2022)]
    years_int=[i for i in range(2015,2022)]
    years = set(years_txt+years_int)
    cruce = set(tabla.columns).intersection(set(years))
    if len(cruce) > 0: col_anyos=True
    
    # Si no son anyos las columnas, buscamos filas que sean anyos:
    contexto = None
    if not col_anyos:
        for i in tabla.iterrows():
            #print(i[1].values)
            try:
                cruce = set(i[1].values).intersection(set(years))
            except:
                cruce=[]
            if len(cruce)>0: # Si encontramos una fila con anyos:
                # Asignamos los anyos a las columnas:
                tabla.columns = i[1].values
                try: 
                    contexto = i[1].name
                except:
                    contexto = None
                # Drop de la fila:
                tabla.drop(i[0], inplace=True)
                col_anyos=True
                break
    
    # Si las columnas son anyos y hemos pasado un anyo actual, nos lo quedamos
    if col_anyos and anyo_actual:
        tabla = tabla[tabla.columns[tabla.columns==anyo_actual]]
        
    
    # Procesos anteriores pueden haber dejado filas vacías, las eliminamos:
    tabla.replace(nan_equiv, np.nan, inplace=True)
    tabla.dropna(axis=0, how='all', inplace=True)
    tabla.dropna(axis=1, how='all', inplace=True)
    # Pasamos a texto:
    texto = ''
    for i in tabla.items():
        # Lista con cada combinatoria columna-fila-valor:
        txt = [f' {titulo} + {i[0]} + {x[0]} = {x[1]}; '
               for x in list(i[1].items())]
        
        # Pasamos a texto:
        add= ''.join(txt)
        if contexto:
            txt = [f' {titulo} + {contexto} + {i[0]} + {x[0]} = {x[1]}; '
                   for x in list(i[1].items())]
            
            add = ''.join(txt)
        
        add = add.replace('  ',' ').replace('\n','; ').replace('  ','')
        
        texto += f';  Tabla={titulo}: {add}'
    return texto

def extract_text_from_pdf(pdf_path) -> list:
    '''
    Función para extraer texto de un pdf y limpiarlo.
    Devuelve una lista de str, cada una es una página del pdf.
    '''
    # Abrimos el pdf
    #with open(pdf_path, 'rb') as f:
    f = pdf_path
    pdf = pypdf.PdfReader(f)
    # Obtenemos el número de páginas
    num_pags = len(pdf.pages)
    count = 0
    text = []
    # Iteramos sobre las páginas
    for pag in pdf.pages:
        count +=1
        texto_pagina = pag.extract_text()
        tablas = tabula.read_pdf(pdf_path, pages=count)
        for tabla in tablas:
            texto_pagina += f'; {tabla_a_texto(tabla=tabla)}; '
        texto_pagina = limpieza_texto(texto_pagina)
        text.append(texto_pagina)
    return text

#############################
# Langchain query motor
#############################
def save_text_pdf_to_csv(save_path: str, text: list):
    '''
    Función para guardar el texto procesado de un pdf en un csv.
    Lo hace de tal forma que cada hoja del doc sea una fila en csv.
    '''
    text = text.copy()
    with open(save_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([[str(i)] for i in text])
    return

def contador_tokens(texto, tokenizador= tiktoken.get_encoding('cl100k_base')):
    return len(tokenizador.encode(texto, disallowed_special=()))

def create_index_from_csv(csv_path: str):
    '''
    Función para crear un índice de búsqueda de un csv.
    '''
    # Cargamos el csv
    loader = CSVLoader(file_path=csv_path)
    # Creamos el índice
    index = VectorstoreIndexCreator().from_loaders([loader])
    return index
    



#############################
# OpenAI embeddings & queries
#############################
'''
DEPRECATED
El objetivo de esta sección era noble, pero en el mundo actual y a
la velocidad que dan las cosas era evidente que esto iba a ser 'mucho trabajo'
OBVIAMENTE esto ya se ha implementado de manera directa en otras librerías.
Es por esto que se usa e implementa el bloque 'LANGCHAIN'.
Se deja el código como un homenaje a la idea original.
'''
'''
Llamadas a la API de OpenAI para obtener embeddings y hacer queries.
Funciones simples que dado un texto devuelven embeddings y dado un input
con contexto te devuelve una respuesta del modelo GPT-3.5.
'''
# EMBEDDINGS
# ToDo:Revisar y optimizar. Contar con límites de la API.
def get_embeddings(text: str, model: str = "ada") -> np.array:
    '''
    Función para obtener los embeddings de un texto.
    '''
    # Uso de API de OpenAI para obtener embeddings
    response = openai.Embedding.create(
        model=model,
        query=text,
    )
    # Devolvemos los embeddings
    return np.array(response['embedding'])

def chop_text(text: list, size:str=1000, overlap: int=50):
    splitter = langchain.text_splitter.RecursiveCharacterTextSplitter(
        chunk_size=size,
        overlap=overlap,
        length_function = len)
    
    docs = splitter.create_documents(text,
                                     metadatas=[
                                         {'pag': i} 
                                         for i in list(range(len(text)))])
    return docs

def vector_store_embeddings(docs: list, model: str = "ada"):
    # Creamos el índice
    embeddings = [get_embeddings(doc.page_content, model=model)
                  for doc in docs]
    store = langchain.vectorstores.Chroma.from_documents(docs,
                                                         embeddings)
    return store

# MENSAJE GPT-3.5:
def send_message(message_log,
                 max_tokens: int = 3800,
                 temp: float = 0.7,
                 stop=None,
                 model: str = "gpt-3.5-turbo",
                 full_output: bool = False):
    '''
    Función para enviar un mensaje a GPT junto con contexto y obtener la
    respuesta del chatbot.

    Parameters
    ----------
    message_log : LIST OF DICTIONARIES
        CONTEXTO: Contexto a proporcionar o historial de conversación,
        como una lista de diccionarios. Cada diccionario debe tener un rol
        y un contenido. El rol puede ser "usuario" o "sistema". El contenido
        es el texto que se envía al modelo.
    max_tokens : INT, optional
        Número tope de tokens. The default is 3800.
    temp : FLOAT, optional
        Temperatura, parámetro que delimita la creatividad del modelo.
        The default is 0.7.
    stop : TYPE, optional
        DESCRIPTION. The default is None.
    model : TYPE, optional
        DESCRIPTION. The default is "gpt-3.5-turbo".
    full_output : BOOL, optional
        Booleano que determina si se devuelve toda la respuesta del server.
        The default is False.

    Returns
    -------
    STRING or DICT
        Devuelve la respuesta del chatbot o el diccionario completo si
        full_output = True.

    '''
    # Uso de API de OpenAI para enviar mensaje y obtener respuesta
    response = openai.ChatCompletion.create(
        model=model,
        messages=message_log,
        max_tokens=max_tokens,
        stop=stop,
        temperature=temp,
    )

    if full_output: return response
    # Si queremos solo la respuesta, filtramos el diccionario
    for choice in response.choices:
        if "text" in choice:
            return choice.text

    # Si no hay respuesta, devolvemos el primer mensaje
    return response.choices[0].message.content

#############
# Orquestador
#############
'''
Organización de las funciones principales.
Principalmente se codifica en embeddings el texto de los pdfs, se hacen
búsquedas por Similitud Coseno y se devuelven N resultados mas cercanos.
Esos resultados (en texto) son los que se le dan como contexto al modelo
gpt-3.5 para que genere la extracción de datos precisos.
'''
