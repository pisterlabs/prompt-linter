import os
import re
import time
from datetime import datetime, timedelta
import textwrap

import cohere
import fitz  # PyMuPDF
import openai
import streamlit as st
from transformers import pipeline
import PyPDF2

# Importaciones específicas del primer código
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from htmlTemplates import css, bot_template, user_template


# Obtener la clave API de OpenAI desde una variable de entorno
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = openai_api_key

# Obtener la clave API de Cohere desde una variable de entorno
cohere_api_key = os.environ.get("COHERE_API_KEY")
co = cohere.Client(cohere_api_key)


# Otras configuraciones iniciales
API_CALLS_LIMIT = 5
TIME_WINDOW = timedelta(minutes=1)
api_calls = []
MODELS_LIST = ["sshleifer/distilbart-cnn-12-6", "facebook/bart-large-cnn", "t5-base", "t5-large", "google/pegasus-newsroom", "Cohere", "gpt-3.5-turbo"]

# Funciones de procesamiento de texto y generación de resumen
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def convertir_pdf_a_chunks(uploaded_file, tamano_chunk=5000):
    doc = fitz.open(stream=uploaded_file.read())
    texto_completo = ""
    for pagina in doc:
        texto_completo += pagina.get_text()
    return texto_completo

def get_text_chunks(text, chunk_size=1000):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def filtrar_por_palabras_clave(texto, keywords):
    sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', texto)
    filtered_sentences = [sentence for sentence in sentences if any(keyword.lower() in sentence.lower() for keyword in keywords)]
    return ' '.join(filtered_sentences)

# Funciones para generar resúmenes y responder preguntas
def generar_resumen_openai(texto):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": texto}]
        )
        # La respuesta se encuentra en el último mensaje generado por el modelo
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def generar_resumen(co, texto):
    global api_calls
    now = datetime.now()
    api_calls = [call for call in api_calls if now - call < TIME_WINDOW]
    while len(api_calls) >= API_CALLS_LIMIT:
        time.sleep(1)
        now = datetime.now()
        api_calls = [call for call in api_calls if now - call < TIME_WINDOW]

    try:
        response = co.summarize(
          text=texto,
          length='auto',
          format='auto',
          model='command',  # Asegurar de que este modelo soporta el español, si es necesario, cámbiarlo por uno que lo haga.
          additional_command='',
          temperature=0.8,
        )
        api_calls.append(now)
        return response.summary
    except cohere.CohereAPIError as e:
        st.error(f"Error específico de Cohere: {e.message}")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# Interfaz de usuario unificada
def main():
    texto_chunks = []

    # Encabezado
    st.image("https://d1b4gd4m8561gs.cloudfront.net/sites/default/files/inline-images/brc-principal_1.png", width=400)
    st.title('Muisca')
    st.subheader('Modelo Unificado de Inteligencia Supervisada:')
    st.write('Computación y Aplicación')

    # Barra lateral
    with st.sidebar:
        # Opción para subir documentos
        uploaded_file = st.file_uploader("Elige un archivo .txt o .pdf", type=["txt", "pdf"])

        # Opciones de configuración para la generación de resúmenes
        model = st.sidebar.selectbox("Elige un modelo", MODELS_LIST)
        scale_percentage = st.sidebar.slider('Ajuste de escala %', min_value=1, max_value=100, value=50)
        chunk_size = st.sidebar.slider('Tamaño del Chunk', min_value=100, max_value=1000, value=500)
        keywords_input = st.text_input("Ingresa las palabras claves (separadas por comas)")
        keywords = [keyword.strip() for keyword in keywords_input.split(",")]

    # Procesamiento de documentos cargados
    if uploaded_file is not None:
        # Obtención del texto de los documentos cargados
        if uploaded_file.type == "application/pdf":
            texto_completo = convertir_pdf_a_chunks(uploaded_file)
        else:
            texto_completo = uploaded_file.read().decode('utf-8')

        # Filtrado del texto por palabras clave
        filtered_text = filtrar_por_palabras_clave(texto_completo, keywords)

        # Obtención de fragmentos de texto
        texto_chunks = get_text_chunks(filtered_text, chunk_size=chunk_size)

        # Preparación para responder preguntas
        vectorstore = get_vectorstore(texto_chunks)
        st.session_state.conversation = get_conversation_chain(vectorstore)

    # Sección para hacer preguntas
    user_question = st.text_input("Haz una pregunta sobre tus documentos:")
    if user_question and 'conversation' in st.session_state:
        handle_userinput(user_question)

    # Sección para mostrar resúmenes
    # Sección para mostrar resúmenes
    if st.button('Resumir') and texto_chunks:
        summarized_text = ""
        resumen_count = 0  # Añade un contador para llevar la cuenta de los resúmenes generados

        if model in MODELS_LIST:
            for i, chunk in enumerate(texto_chunks):
                if resumen_count >= 3:  # Detente si ya has generado tres resúmenes
                    break

                if model == "gpt-3.5-turbo":
                    resumen = generar_resumen_openai(chunk)
                else:
                    resumen = generar_resumen(co, chunk)

                if resumen:
                    summarized_text += resumen + " "
                    st.markdown(f"**Resumen {i+1}:**")
                    st.write(resumen)
                    resumen_count += 1  # Incrementa el contador cada vez que se genera un resumen
        else:
            summarizer = pipeline('summarization', model=model)
            for chunk in texto_chunks:
                chunk_length = len(chunk.split())
                min_length_percentage = max(scale_percentage - 10, 1)
                max_length_percentage = min(scale_percentage + 10, 100)
                min_length = max(int(chunk_length * min_length_percentage / 100), 1)
                max_length = int(chunk_length * max_length_percentage / 100)
                summarized = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                summarized_text += summarized[0]['summary_text'] + " "
        st.text_area('Resumen del texto', summarized_text, height=200)


if __name__ == '__main__':
    main()
