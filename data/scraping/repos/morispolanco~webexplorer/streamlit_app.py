import streamlit as st
import openai

# Configurar la API de OpenAI
openai.api_key = st.text_input("Ingrese su API Key de OpenAI")

def analyze_website(url):
    # Realizar el análisis utilizando GPT-3
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Analiza la URL: {url}.",
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5
    )
    
    # Obtener la respuesta generada por GPT-3
    analysis = response.choices[0].text.strip()
    
    # Mostrar el informe en la aplicación de Streamlit
    st.write("Informe de análisis del sitio web:")
    st.write(analysis)

# Configurar la interfaz de la aplicación de Streamlit
st.title("Analizador de sitios web")
url = st.text_input("Ingrese la URL del sitio web")

# Verificar si se ha ingresado una URL
if url:
    # Verificar si se ha ingresado una API Key
    if openai.api_key:
        # Llamar a la función de análisis del sitio web
        analyze_website(url)
    else:
        st.write("Por favor, ingrese su API Key de OpenAI")
