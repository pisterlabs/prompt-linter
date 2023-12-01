from moviepy.editor import *
import http.client
import json
import re
import jwt
from time import time
import gdown
import os

import requests

import dateutil.parser
from tqdm import tqdm
import base64
from urllib.parse import urljoin


import logging
from typing import TypeVar, cast, Dict, List
from .drive_api import DriveAPI
from .drive_api_exception import DriveAPIException


from pydub import AudioSegment
import openai




#token = "00"#"eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOm51bGwsImlzcyI6IlVFZDBaYV9lVHZxMkFDMWZDNUUtZFEiLCJleHAiOjE2Nzk0MTM4NzQsImlhdCI6MTY3ODgwOTA3NH0.ukP8Ja05WXgbvC-_UgmJF5kh6R_RQ5qUOCmjAiV6eE0"
API_SECRET = os.environ ["API_SECRET_ZOOM"]#"p2juMvG4ifA9x8StadY1lixePaH7Z7nMQuNy"
API_KEY = os.environ["API_KEY_ZOOM"]#"UEd0Za_eTvq2AC1fC5E-dQ"
USUARIO = os.environ["USER_ZOOM"]#"servidor.genie@gmail.com" 
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]






def request_zoom(method, url, payload=None, body=None, exp=0):
        base_url = 'https://api.zoom.us/v2'
        # Replace with your Zoom API credentials
        client_id = 'LUGBeKh_Q8aETAvtgb0IYw'
        client_secret = 'YGMh2LrvizqipxXHzSUfZ6vl2AZU4TT8'
        account_id = 'Fgrn1c2QTgC9mhqtk9xIOQ'

        # Construct the Authorization header
        auth_header = base64.b64encode((client_id + ':' + client_secret).encode()).decode('utf-8')

        # Construct the request data for obtaining access token
        token_data = {
            'grant_type': 'account_credentials',
            'account_id': account_id,
        }

        headers = {
            'Host': 'zoom.us',
            'Authorization': 'Basic ' + auth_header,
            'Content-Type': 'application/x-www-form-urlencoded',
        }

        # Make the POST request to obtain access token
        token_response = requests.post('https://zoom.us/oauth/token', data=token_data, headers=headers)

        # Parse the token response
        if token_response.status_code == 200:
            token_data = token_response.json()
            access_token = token_data.get('access_token')

            # Construct headers for API requests
            api_headers = {
                'Authorization': 'Bearer ' + access_token,
                'Content-Type':'application/json'
            }

            full_url = urljoin(base_url, url)

            if method == 'GET':
                response = requests.get(full_url, headers=api_headers)
            elif method == 'POST':
                response = requests.post(full_url, json=body, headers=api_headers)
            elif method == 'DELETE':
                response = requests.delete(full_url, headers=api_headers)
            else:
                return None, None, 'Invalid method'


            return response.json(), response.status_code,  access_token

        else:
            return None, None, 'Authentication failed'

def obt_video_evento(meeting):
        print(API_SECRET)
        print("variable")
        print(API_KEY)
        res, status, token = request_zoom("GET", f"/v2/meetings/{meeting}/recordings")
        full_filename = f"{meeting}.mp4"
 

        if (status // 100) == 2:
            url = res['recording_files'][0].get('download_url')
            if isinstance(token,str):
                token_decode = token
            else:
                token_decode = token.decode()
            url = f'{url}?access_token={token_decode}'
            url = requests.head(url).headers['Location']
            
            
            response = requests.get(url, stream=True)
            block_size = 32 * 1024  # 32 Kibibytes
            total_size = int(response.headers.get('content-length', 0))
            try:
                t = tqdm(total=total_size, unit='iB', unit_scale=True)
                with open(full_filename, 'wb') as fd:
                    # with open(os.devnull, 'wb') as fd:  # write to dev/null when testing
                    for chunk in response.iter_content(block_size):
                        t.update(len(chunk))
                        fd.write(chunk)  # write video chunk to disk
                print("Descarga Finalizada")
                t.close()
                return True
            except Exception as e:
                # if there was some exception, print the error and return False
                print(e)
                return False
        else:
            return False


def obt_audio_evento(meeting):
        print(API_SECRET)
        print("variable")
        print(API_KEY)
        res, status, token = request_zoom("GET", f"/v2/meetings/{meeting}/recordings")
        full_filename = f"{meeting}.m4a"
        if status>=400:  
            if (status // 100) == 2:
                
                for file in range(len(res['recording_files'])):
                    if (res['recording_files'][file].get('recording_type')=='audio_only'):
                        url = res['recording_files'][file].get('download_url')
                        if isinstance(token,str):
                            token_decode = token
                        else:
                            token_decode = token.decode()
                        url = f'{url}?access_token={token_decode}'
                        url = requests.head(url).headers['Location']
                        
                        
                        response = requests.get(url, stream=True)
                        block_size = 32 * 1024  # 32 Kibibytes
                        total_size = int(response.headers.get('content-length', 0))
                        print("Descarga Finalizada1")
                        break
                try:
                    t = tqdm(total=total_size, unit='iB', unit_scale=True)
                    with open(full_filename, 'wb') as fd:
                        # with open(os.devnull, 'wb') as fd:  # write to dev/null when testing
                        for chunk in response.iter_content(block_size):
                            t.update(len(chunk))
                            fd.write(chunk)  # write video chunk to disk
                    print("Descarga Finalizada1")
                    t.close()
                    return True
                except Exception as e:
                    # if there was some exception, print the error and return False
                    print(e)
                    return False
            else:
                return False
        return False



def upload(id_reunion):
    drive_api = DriveAPI("credenciales-cta-servicio.json","/tmp")  # This should open a prompt.
    try:
        
        # Get url from upload function.
        file_url = drive_api.upload_file(f"{id_reunion}.mp4",f"{id_reunion}.MP4" ,"1PoFVsTKGO7aL9Gm380GWN-HQW6KnTHSX")

        # The formatted date/time string to be used for older Slack clients
        # fall_back = f"{file['date']} UTC"

        # Only post message if the upload worked.
        # message = (f'The recording of _{file["meeting"]}_ on '
        #             "_<!date^" + str(file['unix']) + "^{date} at {time}|" + fall_back + ">_"
        #             f' is <{file_url}| now available>.')
        print(f"Listo la carga de la Reunión {id_reunion}")

    except DriveAPIException as e:
        raise e
    
def upload_text(id_reunion):
    drive_api = DriveAPI("credenciales-cta-servicio.json","/tmp")  # This should open a prompt.
    try:
        
        # Get url from upload function.
        file_url = drive_api.upload_file(f"{id_reunion}.txt",f"{id_reunion}.TXT" ,"1TQhEwLGJmXsoOZ4FY818nGIJf_cH9C3Y")
        # The formatted date/time string to be used for older Slack clients
        # fall_back = f"{file['date']} UTC"

        # Only post message if the upload worked.
        # message = (f'The recording of _{file["meeting"]}_ on '
        #             "_<!date^" + str(file['unix']) + "^{date} at {time}|" + fall_back + ">_"
        #             f' is <{file_url}| now available>.')
        print(f"Listo la carga de la transcripcion analizada de la reunión {id_reunion}")

    except DriveAPIException as e:
        raise e





def convertir_video_a_mp3(video_archivo):

    """
    Convierte un archivo de video a un archivo de audio MP3.

    Args:
        video_archivo (str): Nombre del archivo de video de entrada.
        audio_archivo (str): Nombre del archivo de audio de salida (MP3).

    Returns:
        None
    """
    try:
        audio_archivo = f"{video_archivo}.mp3"
        video_archivo = f"{video_archivo}.MP4"
        
        # Carga el video
        video = VideoFileClip(video_archivo)

        # Extrae el audio del video
        audio = video.audio

        # Guarda el audio en formato MP3
        audio.write_audiofile(audio_archivo)

        # Cierra los archivos
        audio.close()
        video.close()

        print(f"Archivo de audio '{audio_archivo}' creado con éxito.")
    except Exception as e:
        print(f"Error al convertir el video a MP3: {str(e)}")




def eliminar_archivo(archivo):
    archivo = f"{archivo}.mp3"
    """
    Elimina un archivo del sistema.

    Args:
        archivo (str): Nombre del archivo a eliminar.

    Returns:
        None
    """
    try:
        import os
        os.remove(archivo)
        print(f"Archivo '{archivo}' eliminado con éxito.")
    except Exception as e:
        print(f"Error al eliminar el archivo: {str(e)}")



def convert_m4a_to_mp3(source_file, output_file):
    audio = AudioSegment.from_file(source_file, format="m4a")
    audio.export(output_file, format="mp3")

def split_and_transcribe(input_audio_path,extension):


    # Verificar la extensión y realizar la conversión si es necesario
    if extension == '.m4a':
        # Convertir a mp3 si la extensión es m4a
        convert_m4a_to_mp3(f"{input_audio_path}.m4a", f"{input_audio_path}.mp3")
        song = AudioSegment.from_file(f"{input_audio_path}.mp3")
    elif extension == '.mp3':
        # No es necesario convertir, cargar directamente si la extensión es mp3
        song = AudioSegment.from_file(f"{input_audio_path}.mp3")
    else:
        # Extensión no válida
        raise ValueError("Extensión de archivo no compatible")


    # Obtener la duración total del audio en milisegundos
    total_duration = len(song)

    # Calcular la cantidad de segmentos necesarios
    chunk_duration_ms = 10 * 60 * 1000  # 10 minutos en milisegundos
    num_segments = total_duration // chunk_duration_ms

    # Crear el directorio de salida si no existe
    os.makedirs("audio_chunks", exist_ok=True)

    # Inicializar una variable para el texto de transcripción
    full_transcript = ""

    for i in range(num_segments):
        start_time = i * chunk_duration_ms
        end_time = (i + 1) * chunk_duration_ms
        segment = song[start_time:end_time]

        # Exportar el segmento a un archivo temporal
        temp_audio_file = f"{input_audio_path}_segment__{i + 1}.mp3"
        segment.export(temp_audio_file, format="mp3")

        # Transcribir el segmento
        openai.api_key = OPENAI_API_KEY
        audio_file = open(temp_audio_file, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        texto_normal = json.loads(json.dumps(transcript))["text"]
        
        print(f"Transcripción para {temp_audio_file}, archivo {str(i)} de {str(num_segments)}")

        # Cerrar el archivo de audio temporal
        audio_file.close()
        # Agregar la transcripción al texto completo
        full_transcript += f" {transcript}\n\n"

        # Eliminar el archivo de audio temporal
        os.remove(temp_audio_file)

    # Guardar el texto completo de transcripción en un archivo
    output_txt_path = f"{input_audio_path}_transcript.txt"
    with open(output_txt_path, "w") as txt_file:
        txt_file.write(full_transcript)
    return full_transcript


def openai_analyze(transcription):
    openai.api_key = OPENAI_API_KEY
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "Eres un consultor/mentor que acaba de concluir una asesoría y desea elaborar un informe detallado de las actividades realizadas durante la sesión."},
            {"role": "user", "content": f"Basándote en la siguiente transcripción de la reunión, por favor, resume de manera concisa, identifica las actividades, compromisos clave, recomendaciones y conclusiones. Transcripción: {transcription}"},
        ]
    )
    return response


def openai_resumen(transcription):
    openai.api_key = OPENAI_API_KEY
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "Estás leyendo la transcripción de una sesión de asesoría"},
            {"role": "user", "content": f"Basándote en la siguiente transcripción resume el texto con el mayor detalle posible. Transcripción: {transcription}"},
        ],
    )
    return response


def dividir_y_analizar_texto(archivo):
    try:
        file_path= f"{archivo}_transcript.txt"
        # Abre el archivo en modo lectura y carga su contenido
        with open(file_path, 'r', encoding='utf-8') as file:
            texto = file.read()
    
        partes = [texto[i:i+12000] for i in range(0, len(texto), 12000)]
        
        resultados_analisis = []
        resultados_texto_completo = ""
        
        for parte in partes:
            # Analiza cada parte con la API de OpenAI
            try:
                resultado_analisis = openai_resumen(parte)
                if "choices" in resultado_analisis and len(resultado_analisis["choices"]) > 0:
                    resultados_analisis.append(resultado_analisis["choices"][0]["message"]["content"])
                    resultados_texto_completo = resultados_texto_completo + resultado_analisis["choices"][0]["message"]["content"]
                    #print(resultado_analisis)
            except Exception as e:
                # Aquí puedes manejar posibles errores, por ejemplo, reintentar o registrar el error.
                print(f"Error al analizar el texto: {e}")
        resultados_analisis = openai_analyze(resultados_texto_completo)

        # Guardar el texto completo de transcripción en un archivo
        output_txt_path = f"{archivo}.txt"
        with open(output_txt_path, "w") as txt_file:
            txt_file.write(resultados_analisis["choices"][0]["message"]["content"])

    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return
def dwl_file_drive(meeting_id):
    drive_api = DriveAPI("credenciales-cta-servicio.json","/tmp")  # This should open a prompt.
    file_response = drive_api.get_drive_recordings(meeting_id)
    print(file_response)
