import openai  # Importa la biblioteca OpenAI, utilizada para realizar operaciones de procesamiento del lenguaje natural.
import typer  # Importa la biblioteca Typer, utilizada para crear interfaces de línea de comando.
from rich import print  # Importa la función print de la biblioteca Rich, utilizada para imprimir texto en la consola con formato.
from rich.table import Table  # Importa la clase Table de la biblioteca Rich, utilizada para crear tablas con formato en la consola.
import speech_recognition as sr  # Importa la biblioteca SpeechRecognition, utilizada para realizar reconocimiento de voz.
import webbrowser  # Importa la biblioteca webbrowser, utilizada para abrir páginas web.
import os  # Importa la biblioteca os, utilizada para realizar operaciones en el sistema operativo.
from datetime import datetime  # Importa la clase datetime de la biblioteca datetime, utilizada para trabajar con fechas y horas.
import pyjokes  # Importa la biblioteca pyjokes, utilizada para generar chistes aleatorios.
import requests  # Importa la biblioteca requests, utilizada para realizar solicitudes HTTP.
import json  # Importa la biblioteca json, utilizada para trabajar con objetos JSON.
import pyowm  # Importa la biblioteca pyowm, utilizada para obtener información meteorológica.
from bs4 import BeautifulSoup  # Importa la clase BeautifulSoup de la biblioteca Beautiful Soup, utilizada para analizar HTML y XML.
import pyttsx3  # Importa la biblioteca pyttsx3, utilizada para convertir texto en voz.
from gtts import gTTS  # Importa la clase gTTS de la biblioteca Google Text-to-Speech, utilizada para convertir texto en voz.
from pydub import AudioSegment  # Importa la clase AudioSegment de la biblioteca PyDub, utilizada para trabajar con archivos de audio.
from pydub.playback import play  # Importa la función play de la biblioteca PyDub, utilizada para reproducir archivos de audio.
import io  # Importa la biblioteca io, utilizada para trabajar con streams de datos.
import tempfile  # Importa la biblioteca tempfile, utilizada para crear archivos temporales.
from geopy.geocoders import Nominatim  # Importa la clase Nominatim de la biblioteca GeoPy, utilizada para obtener información de ubicación.
from geopy.exc import GeocoderTimedOut  # Importa la excepción GeocoderTimedOut de la biblioteca GeoPy, utilizada para manejar errores en la obtención de ubicaciones.
import sounddevice as sd  # Importa la biblioteca sounddevice, utilizada para trabajar con dispositivos de audio.
import pyaudio  # Importa la biblioteca pyaudio, utilizada para grabar y reproducir audio.
from datetime import datetime, timedelta, timezone  # Importa la clase timedelta de la biblioteca datetime, utilizada para trabajar con intervalos de tiempo.
import locale  # Importa la biblioteca locale, utilizada para manejar la configuración regional del sistema operativo.
import zipfile  # Importa la biblioteca zipfile, utilizada para trabajar con archivos ZIP.
import prettytable  # Importa la clase PrettyTable de la biblioteca prettytable, utilizada para crear tablas con formato.
import pyowm  # Importa la biblioteca pyowm, utilizada para obtener información meteorológica.
import unicodedata  # Importa la biblioteca unicodedata, utilizada para trabajar con caracteres Unicode.
import typer  # Importa la biblioteca Typer, utilizada para crear interfaces de línea de comando.
import sounddevice as sd  # Importa la biblioteca sounddevice, utilizada para trabajar con dispositivos de audio.
import pyaudio  # Importa la biblioteca pyaudio, utilizada para grabar y reproducir audio.
import zipfile  # Importa la biblioteca zipfile, utilizada para trabajar con archivos ZIP.
import prettytable  # Importa la clase PrettyTable de la biblioteca prettytable, utilizada para crear tablas con formato.
climacell_api_key = os.getenv('CLIMACELL_API_KEY')  # Obtiene la clave de la API de Climacell.
openai.api_key = os.getenv('API_KEY')  # Configura la clave de la API de OpenAI.
google_news_api_key = os.getenv('GOOGLE_NEWS_API_KEY')  # Obtiene la clave de la API de noticias de Google.

# Función para extraer el archivo ffmpeg.zip. siempre y cuando no existan los archivos ffmpeg.exe y ffprobe.exe en el directorio C:\ffmpeg\bin.
def extraer_ffmpeg():
    directorio_ffmpeg = "C:\\ffmpeg"
    ffmpeg_exe = os.path.join(directorio_ffmpeg, "bin", "ffmpeg.exe")
    ffprobe_exe = os.path.join(directorio_ffmpeg, "bin", "ffprobe.exe")

    if not os.path.exists(ffmpeg_exe) or not os.path.exists(ffprobe_exe):
        with zipfile.ZipFile("ffmpeg.zip", "r") as zip_ref:
            zip_ref.extractall("C:\\")

# Extrae el archivo ffmpeg.zip.
extraer_ffmpeg()

# Validación de las claves de las APIs.
def configurar_api_keys():

    # Valida la clave de la API de OpenAI.
    if not openai.api_key:
        print("No se encontró la API Key de OpenAI.")
        print("1. Obtén una API Key en https://beta.openai.com/signup")
        print("2. Para agregar la clave como variable de entorno en Windows, sigue estos pasos:")
        print("   a. Haz clic derecho en 'Mi PC' o 'Este equipo' y selecciona 'Propiedades'.")
        print("   b. Haz clic en 'Configuración avanzada del sistema' en el panel izquierdo.")
        print("   c. Haz clic en 'Variables de entorno'.")
        print("   d. En 'Variables del sistema', haz clic en 'Nuevo...'.")
        print("   e. Establece el nombre de la variable como 'API_KEY' y su valor como la clave obtenida.")
        print("   f. Haz clic en 'Aceptar' y reinicia cualquier consola o IDE en uso.")
        print("   g. Puedes seguir estas guías para configurar variables de entorno en diferentes sistemas operativos:")
        print("      -. Windows: https://docs.microsoft.com/es-es/powershell/module/microsoft.powershell.core/about/about_environment_variables?view=powershell-7.1")
        print("      -. macOS y Linux: https://www.cyberciti.biz/faq/set-environment-variable-unix/")
    elif not validar_api_key(openai.api_key):
        return False

    # Valida la clave de la API de Google News.
    if not climacell_api_key:
        print("No se encontró la API Key de Climacell.")
        print("1. Obtén una API Key en https://www.tomorrow.io/signup")
        print("2. Para agregar la clave como variable de entorno en Windows, sigue los pasos mencionados en la API Key de OpenAI.")
        print("   a. Establece el nombre de la variable como 'CLIMACELL_API_KEY' y su valor como la clave obtenida.")
        print("   b. Haz clic en 'Configuración avanzada del sistema' en el panel izquierdo.")
        print("   c. Haz clic en 'Variables de entorno'.")
        print("   d. En 'Variables del sistema', haz clic en 'Nuevo...'.")
        print("   e. Establece el nombre de la variable como 'CLIMACELL_API_KEY' y su valor como la clave obtenida.")
        print("   f. Haz clic en 'Aceptar' y reinicia cualquier consola o IDE en uso.")
        print("      -. Windows: https://docs.microsoft.com/es-es/powershell/module/microsoft.powershell.core/about/about_environment_variables?view=powershell-7.1")
        print("      -. macOS y Linux: https://www.cyberciti.biz/faq/set-environment-variable-unix/")
    elif not validar_climacell_api_key(climacell_api_key):
        return False

    # Valida la clave de la API de noticias de Google.
    if not google_news_api_key:
        print("No se encontró la API Key de Google News.")
        print("1. Obtén una API Key en https://newsapi.org/s/google-news-api")
        print("2. Para agregar la clave como variable de entorno en Windows, sigue los pasos mencionados en la API Key de OpenAI.")
        print("   a. Establece el nombre de la variable como 'GOOGLE_NEWS_API_KEY' y su valor como la clave obtenida.")
        print("   b. Haz clic en 'Configuración avanzada del sistema' en el panel izquierdo.")
        print("   c. Haz clic en 'Variables de entorno'.")
        print("   d. En 'Variables del sistema', haz clic en 'Nuevo...'.")
        print("   e. Establece el nombre de la variable como 'GOOGLE_NEWS_API_KEY' y su valor como la clave obtenida.")
        print("   f. Haz clic en 'Aceptar' y reinicia cualquier consola o IDE en uso.")
        print("      -. Windows: https://docs.microsoft.com/es-es/powershell/module/microsoft.powershell.core/about/about_environment_variables?view=powershell-7.1")
        print("      -. macOS y Linux: https://www.cyberciti.biz/faq/set-environment-variable-unix/")
    elif not validar_google_news_api_key(google_news_api_key):
        return False

    return True

# Función para validar la clave de la API de OpenAI.
def validar_api_key(api_key):
    if not api_key.startswith('sk-'):
        print(f"Error: la clave 'api_key' no comienza con 'sk-'. La clave es {api_key}. Y empieza con {api_key[0:3]}")
        print("1. Obtén una API Key en https://newsapi.org/s/google-news-api")
        decir("Error: la API Key de OpenAI no comienza con 'sk-'. Obtén una API Key en https://newsapi.org/s/google-news-api")
        return False

    if len(api_key) != 51:  # Establece aquí la longitud mínima requerida para api_key
        print(f"Error: la clave 'api_key' tiene menos caracteres de los necesarios. La clave es {api_key} y tiene {len(api_key)} caracteres.")
        print("1. Obtén una API Key en https://newsapi.org/s/google-news-api")
        decir("Error: la API Key de OpenAI tiene menos caracteres de los necesarios. Deben ser 51. Obtén una API Key en https://newsapi.org/s/google-news-api")
        return False

    return True

# Función para validar la API Key de Google News
def validar_google_news_api_key(google_news_api_key):
    if len(google_news_api_key) != 32:  # Establece aquí la longitud mínima requerida para google_news_api_key
        print(f"Error: la clave 'google_news_api_key' tiene menos caracteres de los necesarios. La clave es {google_news_api_key} y tiene {len(google_news_api_key)} caracteres.")
        print("1. Obtén una API Key en https://newsapi.org/s/google-news-api")
        decir("Error: la API Key de Google News tiene menos caracteres de los necesarios. Deben ser 32. Obtén una API Key en https://newsapi.org/s/google-news-api")
        return False

    return True

# Función para validar la API Key de Climacell
def validar_climacell_api_key(climacell_api_key):
    if len(climacell_api_key) != 32:  # Establece aquí la longitud mínima requerida para climacell_api_key
        print(f"Error: la clave 'climacell_api_key' tiene menos caracteres de los necesarios. La clave es {climacell_api_key} y tiene {len(climacell_api_key)} caracteres.")
        print("1. Obtén una API Key en https://www.tomorrow.io/signup")
        decir("Error: la API Key de Climacell tiene menos caracteres de los necesarios. Deben ser 32. Obtén una API Key en https://www.tomorrow.io/signup")
        return False

    return True

# Función para obtener la fecha actual
def obtener_fecha_actual():
    # Establecer la configuración regional en español
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')

    # Obtener la fecha actual en la zona horaria de Bogotá, Colombia
    fecha_actual = datetime.now(timezone(-timedelta(hours=5)))

    # Dar formato a la fecha en el formato deseado (por ejemplo, "20 de enero de 2023")
    fecha_formateada = fecha_actual.strftime("%d de %B de %Y")

    return fecha_formateada

# Función para obtener la hora actual en formato de 12 horas con AM/PM
def obtener_hora_actual():
    # Establecer la configuración regional en español
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')

    # Obtener la hora actual en formato de 12 horas sin AM/PM
    hora_actual = datetime.now().strftime("%I:%M")

    # Obtener la parte de AM/PM de la hora actual
    hora_24 = datetime.now().hour

    # Verificar si es AM o PM y agregar a.m. o p.m. según corresponda
    if hora_24 < 12:
        hora_actual += " a.m."
    else:
        hora_actual += " p.m."

    return hora_actual


# Función para quitar las tildes de una cadena de texto
def quitar_tildes(cadena):
    s = ''.join((c for c in unicodedata.normalize('NFD', cadena) if unicodedata.category(c) != 'Mn'))
    return s


# Inicializa el motor de texto a voz
engine = pyttsx3.init()
tipo_voz = "sintetica"  # Predeterminado

# Función para detectar los dispositivos de audio disponibles
def detectar_dispositivos_audio():
    altavoces_disponibles = False
    microfono_disponible = False

    if len(sd.query_devices()) > 0:
        altavoces_disponibles = True

    p = pyaudio.PyAudio()
    num_microfonos = p.get_device_count()
    for i in range(num_microfonos):
        info_microfono = p.get_device_info_by_index(i)
        if info_microfono["maxInputChannels"] > 0:
            microfono_disponible = True
            break

    return altavoces_disponibles, microfono_disponible

# Función para seleccionar el modo de entrada (audio o escrito)
def seleccionar_modo_entrada():
    _, microfono_disponible = detectar_dispositivos_audio()
    if microfono_disponible:
        return "audio"
    else:
        return "escrito"

# Función para cambiar el tipo de voz (sintetica o natural)
def cambiar_tipo_voz(nuevo_tipo_voz):
    global tipo_voz
    nuevo_tipo_voz_sin_tildes = quitar_tildes(nuevo_tipo_voz.lower())

    if nuevo_tipo_voz_sin_tildes in ["sintetica", "natural"]:
        tipo_voz = nuevo_tipo_voz_sin_tildes
    else:
        raise ValueError("Tipo de voz no válido. Debe ser 'sintetica' o 'natural'.")

# Función para decir algo usando el motor de texto a voz
def decir(texto: str):
    global tipo_voz
    if tipo_voz == "sintetica":
        engine.say(texto)
        engine.runAndWait()
    elif tipo_voz == "natural":
        archivo_temporal = os.path.join(os.getcwd(), "temp_audio.mp3")
        tts = gTTS(texto, lang='es')
        tts.save(archivo_temporal)
        audio = AudioSegment.from_mp3(archivo_temporal)
        play(audio)
        os.remove(archivo_temporal)


# Función para obtener la ubicación geográfica basada en la dirección IP
def obtener_ubicacion_ip():
    ip_api_url = "http://ip-api.com/json/"
    ip_location = requests.get(ip_api_url).json()
    return ip_location["countryCode"]

# Función para buscar noticias en función de la ubicación usando la API de Google News
def buscar_noticias_google_news(country_code):
    google_news_api_key = os.getenv('GOOGLE_NEWS_API_KEY')
    google_news_url = f"https://newsapi.org/v2/top-headlines?country={country_code}&apiKey={google_news_api_key}"
    news_response = requests.get(google_news_url)
    news_data = news_response.json()
    return news_data

# Función para buscar noticias en función de la ubicación usando la API de Climacell
def obtener_clima(coordenadas, clave_api):
    latitud, longitud = coordenadas
    url = f"https://api.tomorrow.io/v4/timelines?location={latitud},{longitud}&fields=temperature&timesteps=1h&units=metric&apikey={clave_api}"
    respuesta = requests.get(url)
    datos = respuesta.json()

    temperatura = datos["data"]["timelines"][0]["intervals"][0]["values"]["temperature"]

    return temperatura

# Función para obtener las coordenadas geográficas de una ciudad
def obtener_coordenadas(ciudad):
    geolocator = Nominatim(user_agent="Voz_GPT3")
    try:
        ubicacion = geolocator.geocode(ciudad)
        if ubicacion:
            coordenadas = (ubicacion.latitude, ubicacion.longitude)
            return coordenadas
        else:
            raise ValueError(f"No se encontraron coordenadas para la ciudad: {ciudad}")
    except GeocoderTimedOut as e:
        raise ValueError("Error: El servicio de geolocalización tardó demasiado en responder.")


def main():
    # Establecer la configuración regional en español
    AudioSegment.converter = "C:\\ffmpeg\\bin\\ffmpeg.exe"
    AudioSegment.ffprobe = "C:\\ffmpeg\\bin\\ffprobe.exe"

    print("[bold green]Asistente de Voz ChatGPT en Python[/bold green]")
    print("[bold green]Versión 1.0[/bold green]")
    print("[bold green]Por: [bold blue]@bladealex9848[/bold blue][/bold green]")
    print("[bold green]Repositorio: https://github.com/bladealex9848/Voice-Assistant-GPT[/bold green]")
    print("[bold green]Licencia: MIT[/bold green]")
    print("[bold green]----------------------------------------------[/bold green]")

    # Tabla de comandos
    table = Table("Comando", "Descripción")
    table.add_row("comandos", "Despliega la lista de comandos disponibles")
    table.add_row("atajos", "Despliega la lista de atajos disponibles")
    table.add_row("salir", "Salir del programa")
    table.add_row("nuevo", "Empezar una nueva conversación")
    table.add_row("hora actual", "Muestra la hora actual")
    table.add_row("fecha actual", "Muestra la fecha actual")
    table.add_row("chiste", "Cuenta un chiste")
    table.add_row("abrir navegador", "Abre el navegador web predeterminado")
    table.add_row("buscar en google (seguido de lo que quieres buscar)", "Busca en Google")
    table.add_row("clima de (seguido de la ciudad)", "Muestra el clima de la ciudad")
    table.add_row("noticias", "Muestra las últimas noticias de tu país de acuerdo a tu ubicación IP")
    table.add_row("voz sintetica", "Cambiar el tipo de voz a sintética")
    table.add_row("voz natural", "Cambiar el tipo de voz a natural")
    table.add_row("cambiar modo", "Cambiar entre modos de entrada de audio y escrito")
    print(table)

    # Tabla de atajos
    table_2 = Table("Atajo de teclado", "Descripción")
    # table_2.add_row("Ctrl + L", "Despliega la lista de comandos disponibles")
    table_2.add_row("Ctrl + C", "Salir del programa")
    # table_2.add_row("Ctrl + N", "Empezar una nueva conversación")
    # table_2.add_row("Ctrl + H", "Muestra la hora actual")
    # table_2.add_row("Ctrl + F", "Muestra la fecha actual")
    # table_2.add_row("Ctrl + J", "Cuenta un chiste")
    # table_2.add_row("Ctrl + O", "Abre el navegador web predeterminado")
    # table_2.add_row("Ctrl + G", "Busca en Google")
    # table_2.add_row("Ctrl + A", "Muestra el clima de la ciudad")
    # table_2.add_row("Ctrl + T", "Muestra las últimas noticias de tu país de acuerdo a tu ubicación IP")
    # table_2.add_row("Ctrl + S", "Cambiar el tipo de voz a sintética")
    # table_2.add_row("Ctrl + V", "Cambiar el tipo de voz a natural")
    # table_2.add_row("Ctrl + M", "Cambiar entre modos de entrada de audio y escrito")
    print(table_2)

    # Detectar dispositivos de audio
    altavoces_disponibles, microfono_disponible = detectar_dispositivos_audio()
    if microfono_disponible:
        modo_entrada = "audio"
    else:
        modo_entrada = "escrito"

    decir(f"Bienvenido a ChatGPT en modo {modo_entrada}. ¿En qué puedo ayudarte?")

    # Contexto del asistente
    context = {"role": "system",
               "content": "Eres un asistente muy útil."}
    messages = [context]

    try:
        while True:
            content = __prompt(modo_entrada).lower()

            # Comando para cambiar el modo de entrada de audio a escrito y viceversa
            if content == "cambiar modo":
                if modo_entrada == "audio":
                    modo_entrada = "escrito"
                else:
                    modo_entrada = "audio"
                decir(f"Modo de entrada cambiado a {modo_entrada}")
                continue

            # Comando para iniciar una nueva conversación con el asistente de voz, borrando el contexto
            if content == "nuevo":
                print("¡Empezamos una nueva conversación!")
                decir("¡Empezamos una nueva conversación!")
                messages = [context]
                content = __prompt(modo_entrada)

            # Comando para mostrar la fecha actual con el formato: día de la semana, día del mes de año
            if content == "fecha actual":
                fecha_actual = obtener_fecha_actual()
                print(f"La fecha actual es: {fecha_actual}")
                decir(f"Hoy es {fecha_actual}")
                continue

            # Comando para mostrar la hora actual con el formato: hora:minutos:segundos AM/PM (12 horas)
            if content == "hora actual":
                hora_actual = obtener_hora_actual()
                print(f"La hora actual es: {hora_actual}")
                decir(f"Son las {hora_actual}")

                continue

            # Comando para contar un chiste usando la librería pyjokes
            if content == "chiste":
                chiste = pyjokes.get_joke(language="es")
                print(f"Chiste: {chiste}")
                decir(chiste)
                continue

            # Comando para abrir el navegador web predeterminado
            if content == "abrir navegador":
                webbrowser.open("https://www.google.com")
                decir("Abriendo navegador web")
                continue

            # Comando para buscar en Google
            if content.startswith("buscar en google"):
                termino_busqueda = content[16:]
                url = f"https://www.google.com/search?q={termino_busqueda}"
                webbrowser.open(url)
                decir(f"Esto es lo que encontré sobre {termino_busqueda}")
                continue

            # Comando para obtener el clima de una ciudad en específico, usando la API de ClimaCell
            if content.startswith("clima de"):
                ciudad = content[8:].strip()

                try:
                    coordenadas = obtener_coordenadas(ciudad)
                    temperatura = obtener_clima(coordenadas, climacell_api_key)
                    print(f"En {ciudad} la temperatura es de {temperatura}°C")
                    decir(f"En {ciudad} la temperatura es de {temperatura}°C")
                except requests.exceptions.RequestException as e:
                    print(f"Error en la solicitud: {e}")
                    decir(f"Error en la solicitud: {e}")
                except Exception as e:
                    print(f"Error inesperado: {e}")
                    decir(f"Error inesperado: {e}")
                continue

            # Comando para obtener noticias de Google News
            if content.startswith("noticias"):
                country_code = obtener_ubicacion_ip()
                news_data = buscar_noticias_google_news(country_code)

                # Imprimir las noticias
                if news_data["status"] == "ok":
                    table_1 = prettytable.PrettyTable(["Título", "Fuente", "Fecha"])
                    table_1.align["Título"] = "l"
                    table_1.align["Fuente"] = "l"
                    table_1.align["Fecha"] = "l"
                    for article in news_data["articles"]:
                        title = article["title"]
                        source = article["source"]["name"]
                        date = datetime.strptime(article["publishedAt"], "%Y-%m-%dT%H:%M:%SZ").strftime("%d/%m/%Y")
                        table_1.add_row([title, source, date])
                    print(table_1)
                    decir("Estas son las noticias más relevantes del momento.")
                else:
                    print("Error al obtener noticias de Google News")
                    decir("Error al obtener noticias de Google News")
                continue

            # Comandos de voz para salir del programa
            if content == "salir":
                # exit = typer.confirm("¿Estás seguro de que quieres salir?")
                # if exit:
                print("¡Hasta pronto!")
                decir("¡Hasta pronto!")
                return

                return __prompt(modo_entrada)

            # Cambiar el tipo de voz
            if content.startswith("voz "):
                nuevo_tipo_voz = content[4:].strip()
                try:
                    if nuevo_tipo_voz == "natural":
                        if os.path.isfile("C:\\ffmpeg\\bin\\ffmpeg.exe"):
                            # Validar si la carpeta 'C:\ffmpeg\bin' está en la variable de entorno Path
                            paths = os.environ['PATH'].split(';')
                            if "C:\\ffmpeg\\bin" in paths:
                                cambiar_tipo_voz(nuevo_tipo_voz)
                                print(f"Tipo de voz cambiado a: {tipo_voz}")
                                decir(f"Tipo de voz cambiado a: {tipo_voz}")
                            else:
                                print("La carpeta 'C:\\ffmpeg\\bin' no está en la variable de entorno Path, se mantendrá la voz sintética")
                                decir("La carpeta 'C:\\ffmpeg\\bin' no está en la variable de entorno Path, se mantendrá la voz sintética")
                                print("Para agregar la carpeta 'C:\\ffmpeg\\bin' a la variable de entorno Path en Windows, sigue estos pasos:")
                                print("1. Abre el menú de inicio y busca 'Sistema' en la barra de búsqueda. Haz clic en 'Editar la variable de entorno del sistema'.")
                                print("2. En la ventana Propiedades del sistema, haz clic en el botón 'Variables de entorno'.")
                                print("3. Busca la variable 'Path' en la sección 'Variables del sistema' y haz clic en 'Editar'.")
                                print("4. Haz clic en 'Nuevo' y escribe 'C:\\ffmpeg\\bin' (sin comillas).")
                                print("5. Haz clic en 'Aceptar' para guardar los cambios y cierra todas las ventanas.")
                                print("6. Vuelve a intentar cambiar el tipo de voz a natural en el programa y verifica si ahora se encuentra la carpeta 'C:\\ffmpeg\\bin' en la variable de entorno Path.")
                            continue
                        else:
                            print("No se encontró el archivo 'C:\\ffmpeg\\bin\\ffmpeg.exe'")
                            decir("No se encontró el archivo 'C:\\ffmpeg\\bin\\ffmpeg.exe'")
                            if os.path.isfile("ffmpeg.zip"):
                                with zipfile.ZipFile("ffmpeg.zip", "r") as zip_ref:
                                    zip_ref.extractall("C:\\")
                                    print("El archivo 'ffmpeg.zip' se ha descomprimido en C:\\")
                                    decir("El archivo 'ffmpeg.zip' se ha descomprimido en C:\\")
                            else:
                                print("No se encontró el archivo 'ffmpeg.zip'")
                                decir("No se encontró el archivo 'ffmpeg.zip'")
                            if os.path.isfile("C:\\ffmpeg\\bin\\ffmpeg.exe"):
                                # Validar si la carpeta 'C:\ffmpeg\bin' está en la variable de entorno Path
                                paths = os.environ['PATH'].split(';')
                                if "C:\\ffmpeg\\bin" in paths:
                                    cambiar_tipo_voz(nuevo_tipo_voz)
                                    print(f"Tipo de voz cambiado a: {tipo_voz}")
                                    decir(f"Tipo de voz cambiado a: {tipo_voz}")
                                else:
                                    print("La carpeta 'C:\\ffmpeg\\bin' no está en la variable de entorno Path, se mantendrá la voz sintética")
                                    decir("La carpeta 'C:\\ffmpeg\\bin' no está en la variable de entorno Path, se mantendrá la voz sintética")
                                    print("Para agregar la carpeta 'C:\\ffmpeg\\bin' a la variable de entorno Path en Windows, sigue estos pasos:")
                                    print("1. Abre el menú de inicio y busca 'Sistema' en la barra de búsqueda. Haz clic en 'Editar la variable de entorno del sistema'.")
                                    print("2. En la ventana Propiedades del sistema, haz clic en el botón 'Variables de entorno'.")
                                    print("3. Busca la variable 'Path' en la sección 'Variables del sistema' y haz clic en 'Editar'.")
                                    print("4. Haz clic en 'Nuevo' y escribe 'C:\\ffmpeg\\bin' (sin comillas).")
                                    print("5. Haz clic en 'Aceptar' para guardar los cambios y cierra todas las ventanas.")
                                    print("6. Vuelve a intentar cambiar el tipo de voz a natural en el programa y verifica si ahora se encuentra la carpeta 'C:\\ffmpeg\\bin' en la variable de entorno Path.")
                            else:
                                print("No se pudo encontrar 'C:\\ffmpeg\\bin\\ffmpeg.exe', se mantendrá la voz sintética")
                                decir("No se pudo encontrar 'C:\\ffmpeg\\bin\\ffmpeg.exe', se mantendrá la voz sintética")
                                continue
                    else:
                        cambiar_tipo_voz(nuevo_tipo_voz)
                        print(f"Tipo de voz cambiado a: {tipo_voz}")
                        decir(f"Tipo de voz cambiado a: {tipo_voz}")
                        continue
                except ValueError as e:
                    print(e)
                    decir("Error al cambiar el tipo de voz")
                    continue

            # Comando para ver los comandos
            if content == "comandos":
                print(table)
                decir("Estos son los comandos disponibles")
                continue

            # Comando para ver los atajos
            if content == "atajos":
                print(table_2)
                decir("Estos son los atajos disponibles")
                continue

            # Implementa más comandos aquí

            # Si el usuario no ingresa un comando, se envía el mensaje al asistente
            messages.append({"role": "user", "content": content})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )

            # Si el asistente no entiende el mensaje, se le pregunta al usuario
            response_content = response.choices[0].message.content

            # Los mensajes se almacenan en una lista para que el asistente pueda aprender
            messages.append({"role": "assistant", "content": response_content})

            # Se imprime el mensaje del asistente y se reproduce con la función decir()
            print(f"[bold green]Asistente:[/bold green] {response_content}")
            decir(response_content)

    except KeyboardInterrupt:
        print("\n¡Hasta pronto!")
        decir("¡Hasta pronto!")

def __prompt(modo_entrada: str) -> str:
    if modo_entrada == "audio":
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print(f"[bold green]Por favor, habla:[/bold green]")
            audio = r.listen(source)

            try:
                text = r.recognize_google(audio, language="es-ES")
                print(f"[bold green]Usted dijo:[/bold green] {text}")
                return text
            except sr.UnknownValueError:
                print("No entendí lo que dijiste. Por favor, intenta de nuevo.")
                decir("No entendí lo que dijiste. Por favor, intenta de nuevo.")
                return __prompt(modo_entrada)
            except sr.RequestError as e:
                print(f"Error al obtener resultados; {e}")
                decir("Error al obtener resultados")
                return __prompt(modo_entrada)
    elif modo_entrada == "escrito":
        return input(f"Escribe tu pregunta: ")
    else:
        raise ValueError("Modo de entrada no válido. Debe ser 'audio' o 'escrito'.")

if __name__ == "__main__":
    if configurar_api_keys():
        main()
    else:
        print("No se pudieron configurar las claves API. Saliendo del programa.")