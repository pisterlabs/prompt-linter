import streamlit as st
from PIL import Image
from openai import OpenAI
from pathlib import Path


st.set_page_config(page_title="Streamlit Media", page_icon=":sunglasses:")

st.title("Streamlit Media Elements")

st.header("Veamos como cargar varios elementos multimedia en Streamlit")

st.subheader("Audio")


uploaded_file = st.file_uploader("Sube un archivo de audio", type=['wav', 'mp3', 'ogg'])

# Comprobar si se ha subido un archivo
if uploaded_file is not None:
    # Leer el archivo Excel con Pandas
    st.audio(uploaded_file)
    
else:
    st.write("Por favor, sube un archivode audio")


st.markdown("---")

st.subheader("Video")
video = st.file_uploader("Sube un archivo de video", type=['mp4', 'avi', 'mov'])


st.video(video)

st.markdown("---")

st.subheader("Imágenes")

image = st.file_uploader("Sube un archivo de imagen", type=['png', 'jpg', 'jpeg'])
if image is not None:
    st.image(image, caption='imagen del usuario', use_column_width=True)
    imagen = Image.open(image)
    imagen_en_grises = imagen.convert('L')
    st.image(imagen_en_grises, caption='imagen en grises', use_column_width=True)

else:
    st.warning("Por favor, sube un archivo de imagen (.png, .jpg, .jpeg).")

st.markdown("---")




client = OpenAI(api_key="acá va tu API_kEY de OpenAI que puedes obtener en https://platform.openai.com/api-keys")

audio_file= open("models/data/PruebaAudio.mp3", "rb")
transcript = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file
)

st.write(transcript.text)

st.markdown("---")

st.subheader("Texto a voz")

voice_text = st.text_input("Escribe un texto para convertirlo en voz",value="Hola, soy un texto de prueba, espero te encuentres muy bien")

if voice_text is not None:
    
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=voice_text
    )

    response.stream_to_file(speech_file_path)
    st.audio("pages/speech.mp3")