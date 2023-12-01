import streamlit_authenticator as stauth
import streamlit as st
import yaml
from yaml.loader import SafeLoader
import os
import Historial
import openai
from openai.error import APIError, InvalidRequestError, OpenAIError, APIConnectionError
import pygame
from gtts import gTTS
from gtts.tts import gTTSError
import glob
from PIL import Image
from datetime import datetime
from cryptography.fernet import Fernet
from cryptography.fernet import InvalidToken

# Memoria
if 'paciente_actual' not in st.session_state:
    st.session_state["paciente_actual"] = None

if 'pantalla_inicial' not in st.session_state:
    st.session_state["pantalla_inicial"] = True

if 'fichero' not in st.session_state:
    st.session_state["fichero"] = None

if 'titulo' not in st.session_state:
    st.session_state["titulo"] = None

if 'texto' not in st.session_state:
    st.session_state["texto"] = None

if 'resumen_txt' not in st.session_state:
    st.session_state["resumen_txt"] = None

if 'error_gtts' not in st.session_state:
    st.session_state["error_gtts"] = False

if 'resumen_fichero' not in st.session_state:
    st.session_state["resumen_fichero"] = ""

st.set_page_config(page_title="Asistente voz | Área médicos", page_icon="favicon.png", layout="wide", initial_sidebar_state="collapsed", menu_items=None)

# Cargo el fichero de estilos css
with open('styles_medicos.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def generar_audio(archivo, texto, carpeta, carpeta_paciente, DNI, nombre, apellido, tipo):
    if tipo == "Resumen": ruta = archivo.split("\\")
    else: ruta = archivo.split("\\")

    fichero = ruta[2].split("_")
    fecha = fichero[4]
    ruta_audio = os.path.join(carpeta, carpeta_paciente, f"{tipo}_{DNI}_{nombre}_{apellido}_{fecha}.wav")

    if not os.path.exists(ruta_audio):
        with st.spinner("Se está generando el audio..."):
            try: 
                tts = gTTS(text=st.session_state["resumen_txt"], lang="es", slow=False)
                tts.save(ruta_audio)
                encriptar_audio(ruta_audio)
                tts = gTTS(text=st.session_state["resumen_txt"], lang="es", slow=False)
                tts.save(ruta_audio)
                encriptar_audio(ruta_audio)
                st.session_state["error_gtts"] = False
            except gTTSError:
                msj = "No es posible generar el audio del mensaje, por ahora continuaremos la consulta con el texto."
                st.warning(msj)
                st.session_state["error_gtts"] = True
    return ruta_audio

# Métodos
def reproducir(ruta_audio, boton):
    desencriptar_audio(ruta_audio)
    audio = "desencriptado.wav"
    if boton == "reproduce":
        pygame.mixer.pause()
        sonido = pygame.mixer.Sound(audio)
        sonido.play()
        os.remove(audio)
    
    elif boton == "pausa":
        pygame.mixer.pause()
        os.remove(audio)

def encriptar_audio(audio):
    key = Historial.obtener_key()
    llave = Fernet(key)
    with open(audio, 'rb') as a:
        audio_original = a.read()
    audio_encriptado = llave.encrypt(audio_original)

    with open(audio, 'wb') as a:
        a.write(audio_encriptado)

def desencriptar_audio(audio):
    key = Historial.obtener_key()
    llave = Fernet(key)
    with open(audio, 'rb') as a:
        audio_encriptado = a.read()
    audio_desencriptado = llave.decrypt(audio_encriptado)
    
    with open("desencriptado.wav", 'wb') as a:
        a.write(audio_desencriptado)

def encriptar_fichero(ruta, texto_fichero):
    key = Historial.obtener_key()
    llave = Fernet(key)
    fichero_encriptado = llave.encrypt(texto_fichero)
    with open(ruta, "wb") as f:
        f.write(fichero_encriptado)

def desencriptar_fichero(fichero):
    key = Historial.obtener_key()
    llave = Fernet(key)
    with open(fichero, "rb") as f:
        fichero_encriptado = f.read()
    try:
        fichero_original = llave.decrypt(fichero_encriptado)
        fichero_original = fichero_original.decode("utf-8")
    except InvalidToken:
        fichero_original = ""
    return fichero_original

def busqueda_ficheros(archivo, carpeta, carpeta_paciente, tipo, dni, nombre, apellido, anonimo):
    with open(archivo,encoding='utf-8') as f:
        fichero = archivo.split("_")
        fecha_txt = fichero[6].split(".")
        fecha = fecha_txt[0]

        if tipo == "Gráfica radar": simbolo = "📊"
        else: simbolo = "📄"

        if st.button(f'{simbolo} {tipo}-{fecha}'):
            if tipo == "Gráfica radar":
                st.session_state["pantalla_inicial"] = False
                st.session_state["fichero"] = archivo
                st.session_state["titulo"] = f"{tipo} de {nombre} {apellido} a {fecha}"
                st.session_state["texto"] = False

            else:
                txt_fichero = desencriptar_fichero(archivo)
                if txt_fichero == "":
                    st.error("No tienes permiso para acceder a este fichero.")
                else:
                    st.session_state["pantalla_inicial"] = False
                    st.session_state["fichero"] = txt_fichero
                    st.session_state["texto"] = True
                    if anonimo: 
                        st.session_state["titulo"] = f"Consulta anónima a {fecha}"
                        mostrar_fichero(st.session_state["fichero"],st.session_state["titulo"],True)
                    elif nombre == "Anónimo":
                        st.session_state["titulo"] = f"{tipo} de {dni} a {fecha}"
                    else: st.session_state["titulo"] = f"{tipo} de {nombre} {apellido} a {fecha}"



def pantalla_inicial():
    carpeta = "historiales"
    if not os.path.exists(carpeta):os.mkdir(carpeta)
    ruta = os.path.join(carpeta)

    titulo = '<div class="intro"> Asistente virtual para el triaje de patologías de la voz </div>'
    st.markdown(titulo, unsafe_allow_html=True)

    footer= '<div class="footer">Creado con <i>Streamit</i> por Sara Ibáñez Alloza <br> <b>TFG 2023</b></div>'
    st.markdown(footer,unsafe_allow_html=True)

    col1, col2, col3 = st.columns([6,4,4], gap="medium")
    
    with col1:
        seccion_disponibles = '<div class="seccion_medicos">Búsqueda de pacientes</div>'
        st.markdown(seccion_disponibles, unsafe_allow_html=True)
    
    with col3:
        bienvenida = f'<div class="bienvenido">¡Bienvenid@ <i>{name}</i>!</div>'
        st.markdown(bienvenida, unsafe_allow_html=True)
        authenticator.logout('**Cerrar sesión**👋', 'main')

    
    carpetas = os.listdir(carpeta)
    if len(carpetas) == 0:
        no_pacientes = '<div class="texto_medicos">Todavía ningún paciente ha usado el asistente.</div>'
        st.markdown(no_pacientes, unsafe_allow_html=True)
    else:
        pacientes = {}
        for carpeta_paciente in carpetas:
            paciente = carpeta_paciente.split("_")
            DNI = paciente[0]
            nombre = paciente[1]
            apellido = paciente[2]
            paciente_completo = f"{DNI} {nombre} {apellido}"
            pacientes[paciente_completo] = carpeta_paciente
        opciones = list(pacientes.keys())
        paciente = st.multiselect("buscador", opciones, max_selections=1,label_visibility="collapsed")
        if len(paciente) > 0: 
            st.session_state["paciente_actual"] = paciente[0]
            st.session_state["pantalla_inicial"] = True

    # Cuando el médico clique en un paciente se mostrará el resumen de los ficheros más actuales y el listado de los ficheros disponibles
    if st.session_state["paciente_actual"] != None and st.session_state["paciente_actual"][0] == "0":
        paciente_seccion = f'<div class="seccion_medicos">Pacientes anónimos</div>'
        st.markdown(paciente_seccion, unsafe_allow_html=True)
        texto = """Aquí puede ver las consultas de los pacientes que no han dados sus datos porque solo
        han querido recibir recomendaciones para sus síntomas."""
        anotacion = f'<div class="texto_medicos">{texto}</div>'
        st.markdown(anotacion, unsafe_allow_html=True)
        
        carpeta_paciente = pacientes.get("0 Anónimo ")
        ruta_carpeta = os.path.join(carpeta, carpeta_paciente)
        archivos_consulta = glob.glob(os.path.join(ruta_carpeta, f'*{"consulta"}*'))
        for fichero_consulta in archivos_consulta:
            busqueda_ficheros(fichero_consulta, carpeta, carpeta_paciente, "Anónimo", DNI, nombre, apellido, True)
    
    elif st.session_state["paciente_actual"] != None:
        paciente = st.session_state["paciente_actual"]
        carpeta_paciente = pacientes.get(paciente)
        paciente = carpeta_paciente.split("_")
        DNI = paciente[0]
        nombre = paciente[1]
        apellido = paciente[2]
        paciente_seccion = f'<div class="seccion_medicos">{DNI}, {nombre} {apellido}</div>'
        st.markdown(paciente_seccion, unsafe_allow_html=True)
        resumen, ficheros = st.columns([8,2], gap="medium")
        ruta_carpeta = os.path.join(carpeta, carpeta_paciente)
        
        with ficheros:
            titulo_ficheros = f'<div class="subseccion_medicos">Ficheros para consultar</div>'
            st.markdown(titulo_ficheros, unsafe_allow_html=True)

            archivos_general = glob.glob(os.path.join(ruta_carpeta, f'*{"voz"}*'))
            for fichero_general in archivos_general:
                busqueda_ficheros(fichero_general, carpeta, carpeta_paciente, "General", DNI, nombre, apellido, False)

            archivos_VHI = glob.glob(os.path.join(ruta_carpeta, f'*{"VHI"}*'))
            for fichero_VHI in archivos_VHI:
                if "resumen" not in fichero_VHI:
                    busqueda_ficheros(fichero_VHI, carpeta, carpeta_paciente, "VHI-30", DNI, nombre, apellido, False)

            archivos_consulta = glob.glob(os.path.join(ruta_carpeta, f'*{"consulta"}*'))
            for fichero_consulta in archivos_consulta:
                busqueda_ficheros(fichero_consulta, carpeta, carpeta_paciente, "Consulta", DNI, nombre, apellido, False)

            graficas_radar = glob.glob(os.path.join(ruta_carpeta, f'*{"radar"}*'))
            for radar in graficas_radar:
                busqueda_ficheros(radar, carpeta, carpeta_paciente, "Gráfica radar", DNI, nombre, apellido, False)

        with resumen:
            #Buscamos el archivo general más reciente
            if archivos_general:
                fichero_general_ultimo = max(archivos_general, key=os.path.getmtime)
                txt_fichero_general = desencriptar_fichero(fichero_general_ultimo)

                #Buscamos el archivo VHI más reciente
                if archivos_VHI:   #el paciente ha podido acabar la consulta completando el anterior cuestionario pero no este
                    fichero_VHI_ultimo = max(archivos_VHI, key=os.path.getmtime)
                    st.session_state["resumen_fichero"] = fichero_VHI_ultimo
                    fecha_modificacion = datetime.fromtimestamp(os.path.getmtime(fichero_VHI_ultimo))
                    
                    if "resumen" in fichero_VHI_ultimo:
                        st.session_state["resumen_txt"] = desencriptar_fichero(fichero_VHI_ultimo)

                    else:
                        txt_fichero_VHI = desencriptar_fichero(fichero_VHI_ultimo)

                        with st.spinner('Generando el resumen del paciente, por favor, no toque nada...'):
                            try:
                                prompt = f"""Quiero que resumas dos textos y que entre sus resúmenes haya un salto de línea. El primero es información sobre un paciente, redáctalo en mínimo párrafo. El segundo es una prueba VHI-30, redáctalo en tres párrafos.
                                El primero:
                                {txt_fichero_general}.

                                La prueba:
                                {txt_fichero_VHI}.

                                De esta prueba quiero que redactes en un párrafo un resumen de cada parte, con un salto de línea entre cada párrafo. Las partes son: Funcional (F), Física (P) y Emocional(E). Recuerda que al comienzo se indica lo que significa cada puntuación y que las puntuaciones están después de los dos puntos de cada afirmación. No olvides añadir en los resúmenes la puntuación total de la parte (se indica en el texto como -X puntos- al lado del nombre de la parte) y al terminar la suma de las tres."""
                                response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                {"role": "system", "content": "set language es"},
                                {"role": "system", "content": "Eres un doctor especializado en otorrinolaringología que va a resumir de forma redactada (separando las tres partes) el estado del paciente de una prueba VHI-30."},
                                {"role": "user", "content": prompt}],
                                max_tokens = 700)
                                st.session_state["resumen_txt"] = response['choices'][0]['message']['content']

                                resumen_VHI_ultimo = fichero_VHI_ultimo.replace(".txt", "_resumen.txt")
                                encriptar_fichero(resumen_VHI_ultimo, st.session_state["resumen_txt"].encode())

                            except (APIError, InvalidRequestError, OpenAIError, APIConnectionError, Exception):
                                st.session_state["resumen_txt"] = "No es posible generar el resumen del paciente, inténtelo más adelante.<br>Puede consultar sus ficheros en la derecha."

                elif fichero_general != None:
                    st.session_state["resumen_fichero"] = fichero_general

                    if "resumen" in fichero_general:
                        st.session_state["resumen_txt"] = desencriptar_fichero(fichero_general)
                    else:
                        with st.spinner('Generando resumen del paciente...'):
                            try:
                                prompt = f"Texto a resumir: {txt_fichero_general}."
                                response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                {"role": "system", "content": "set language es"},
                                {"role": "system", "content": "Eres un doctor especializado en otorrinolaringología que va a resumir una consulta."},
                                {"role": "user", "content": prompt}],
                                max_tokens = 500)
                                st.session_state["resumen_txt"] = response['choices'][0]['message']['content']

                                resumen_general_ultimo = fichero_general.replace(".txt", "_resumen.txt")
                                encriptar_fichero(resumen_general_ultimo, st.session_state["resumen_txt"].encode())
                                
                            except (APIError, InvalidRequestError, OpenAIError, APIConnectionError, Exception):
                                st.session_state["resumen_txt"] = "No es posible generar el resumen del paciente, inténtelo más adelante.<br>Puede consultar sus ficheros en la derecha."
                        
            else:
                txt = f'<div class="texto_medicos">{"El paciente no ha llegado a rellenar todos los datos de ningún cuestionario."}</div>'
                st.markdown(txt, unsafe_allow_html=True)
                st.session_state["resumen_txt"] = ""

            if st.session_state["resumen_txt"] != "":
                aviso_controles = '<div class="texto_medicos"><i>Debajo del texto se encuentran los controles para reproducir y pausar el texto.</i></div>'
                st.markdown(aviso_controles, unsafe_allow_html=True)
                resumen = f'<div class="texto_medicos">{st.session_state["resumen_txt"]}</div>'
                st.markdown(resumen, unsafe_allow_html=True)

                ruta_audio = generar_audio(st.session_state["resumen_fichero"], st.session_state["resumen_txt"], carpeta, carpeta_paciente, DNI, nombre, apellido, "Resumen")

                if st.session_state["error_gtts"] == False:
                    espacio1, play, silenciar, espacio2 = st.columns([1,1,1,1], gap="medium")
                    with play:
                        if st.button("Reproducir 🔊" ,key="reproduce", type="secondary"):
                            reproducir(ruta_audio, "reproduce")
                    with silenciar:
                        if st.button("Silenciar 🔇" ,key="para", type="secondary"):
                            reproducir(ruta_audio, "pausa")
            
            scroll='''
                    <style>
                        [data-testid="column"].css-eb3zmp {
                            overflow-y: auto;
                            overflow-x: hidden;
                            height: min (50vh, fit-content);
                        }
                    </style>
                    '''
            st.markdown(scroll, unsafe_allow_html=True)
        
            if st.session_state["pantalla_inicial"] == False:
                mostrar_fichero(st.session_state["fichero"],st.session_state["titulo"],st.session_state["texto"])   
                
def mostrar_fichero(fichero,titulo,texto):
    vacio, boton = st.columns([10,1], gap="medium")
    with vacio:
        st.markdown(f"<div class=titulo_fichero><i><b>{titulo}</b></i></div>", unsafe_allow_html=True)  
    with boton:
        if st.button("**X**", type="primary"):
            st.session_state["pantalla_inicial"] = True
    
    with st.container():
        if texto:
            fichero_mostrar = fichero.replace("Tú:", "Paciente:")
            st.markdown(f"""<div class="fichero">
                            <div class="seccion_fichero">{titulo}</div>
                            <div class="texto_fichero">{fichero_mostrar}</div>
                        </div>""", unsafe_allow_html=True)    
        
        else:
            st.markdown(f"""<div class="seccion_fichero">{titulo}</div>
                            <div class="texto_fichero">En la siguiente gráfica se comparan los resultados del paciente con los estándares sanos (<span style="color:blue">zona azul</span>).</div>
                        """, unsafe_allow_html=True)   
            image = Image.open(fichero)
            st.image(image)

        scroll='''
            <style>
                .container {
                    overflow-y: auto;
                    overflow-x: hidden;
                }
            </style>
            '''
        st.markdown(scroll, unsafe_allow_html=True)
        

#AUTENTIFICACION
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login(f'Asistente virtual para el triaje de patologías de la voz\n', 'main')
footer= '<div class="footer">Creado con <i>Streamit</i> por Sara Ibáñez Alloza <br> <b>TFG 2023</b></div>'
st.markdown(footer,unsafe_allow_html=True)

#MEDICO AUTORIZADO
if authentication_status:
    st.session_state["paciente_actual"] = None
    pygame.init()   #iniciamos el reproductor de audio
    pantalla_inicial()

elif (authentication_status == False):
    st.error('Usuario y/o contraseña incorrecto')