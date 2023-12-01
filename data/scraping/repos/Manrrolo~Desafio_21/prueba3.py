'''
requirements.txt file contents:

langchain==0.0.154
PyPDF2==3.0.1
python-dotenv==1.0.0
streamlit==1.18.1
faiss-cpu==1.7.4
streamlit-extras
'''


import streamlit as st
from dotenv import load_dotenv
import pickle
import os
import glob
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import tempfile
import requests
from IPython.display import Audio, clear_output
from elevenlabs import generate, play, set_api_key, voices, Models
import datetime
import csv
import json


# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model

    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Prompt Engineer](https://youtube.com/@engineerprompt)')
    add_vertical_space(1)
    st.markdown('Modified with 💪 by [Manrrolo](https://manrrolo.github.io/Page/)')
    use_audio = st.checkbox("Usar respuesta de audio", value=False)  # Nueva opción para activar/desactivar audio


load_dotenv()
state = False
# Define las funciones para generar la respuesta de voz
def get_voice_audio(text, voice_id="ErXwobaYiN019PkySvjV"):
    CHUNK_SIZE = 1024
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
      "Accept": "audio/mpeg",
      "Content-Type": "application/json",
      "xi-api-key": "62e2fd73469f1eac0523ff39cd1489d8"
    }

    data = {
      "text": text,
      "model_id" : "eleven_multilingual_v1",
      "voice_settings": {
        "stability": 0.4,
        "similarity_boost": 1.0
      }
    }

    response = requests.post(url, json=data, headers=headers)

    # Save audio data to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
        f.flush()
        temp_filename = f.name

    return temp_filename

def login():
    username = st.text_input("Nombre de usuario", value=st.session_state.get("username", ""))
    password = st.text_input("Contraseña", type="password")

    if st.button("Iniciar sesión"):
        if username == "test" and password == "test":  # Esta es solo una comprobación dummy. Deberías verificar las credenciales de inicio de sesión de manera segura.
            st.session_state["username"] = username
            st.session_state["loggedin"] = True
        else:
            st.write("Las credenciales proporcionadas no son correctas.")
            st.session_state["loggedin"] = False

    # Aquí comienza tu aplicación principal
    if "loggedin" in st.session_state and st.session_state["loggedin"] == True:
        return True




def main():
    st.subheader("Bienvenido a la aplicación")
    st.header("Chat with PDF 💬")
    # Load all existing .pkl files
    vector_stores = {}
    for file in glob.glob("*.pkl"):
        with open(file, "rb") as f:
            vector_stores[file[:-4]] = pickle.load(f)

    choice = st.radio("Do you want to:", ('Upload a new PDF', 'Use an existing .pkl file'))

    if choice == 'Upload a new PDF':
        # upload a PDF file
        pdf = st.file_uploader("Upload your PDF", type='pdf')

        if pdf is not None:
            with st.spinner('Cargando y procesando el PDF...'):
                pdf_reader = PdfReader(pdf)
                
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                    )
                chunks = text_splitter.split_text(text=text)

                store_name = pdf.name[:-4]
                
                if os.path.exists(f"{store_name}.pkl"):
                    with open(f"{store_name}.pkl", "rb") as f:
                        VectorStore = pickle.load(f)
                else:
                    embeddings = OpenAIEmbeddings()
                    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                    with open(f"{store_name}.pkl", "wb") as f:
                        pickle.dump(VectorStore, f)

                # Add the new or updated VectorStore to the dictionary
                vector_stores[store_name] = VectorStore

    elif choice == 'Use an existing .pkl file':
        if vector_stores:
            selected_store = st.selectbox('Select a preprocessed PDF:', list(vector_stores.keys()))
            VectorStore = vector_stores[selected_store]
        else:
            st.write("No se encontraron archivos .pkl preprocesados.")
            return

    # Accept user questions/query
    query = st.text_input("Ask questions about your PDF file:")

    if query:
        with st.spinner('Obteniendo la respuesta...'):
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query+ " Responde en español")
                print(cb)
            if use_audio:  # Si la opción de audio está activada, generar y mostrar audio
                # Convert the response to audio
                audio_file = get_voice_audio(response)
                st.audio(audio_file)
            st.write(response)
            # Log the question, response, timestamp and document name
            with open("log.txt", "a") as log_file:
                log_file.write(f"Time: {datetime.datetime.now()}, Document: {selected_store}, Question: {query}, Response: {response}\n")
            # Log the question, response, timestamp, and document name
            with open("log.csv", "a", newline='') as log_file:
                writer = csv.writer(log_file)
                writer.writerow([datetime.datetime.now(), selected_store, query, response])
            # Log the question, response, timestamp, and document name
            with open("log.json", "a") as log_file:
                log_entry = {"Time": str(datetime.datetime.now()), "Document": selected_store, "Question": query, "Response": response}
                log_file.write(json.dumps(log_entry) + "\n")


            

if __name__ == "__main__":
    if login():
        main()
    else:
        login()