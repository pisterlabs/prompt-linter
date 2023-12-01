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
    add_vertical_space(5)
    st.markdown('Modified with 💪 by [Manrrolo](https://manrrolo.github.io/Page/)')


load_dotenv()

# Define las funciones para generar la respuesta de voz
def get_voice_audio(text, voice_id="ErXwobaYiN019PkySvjV"):
    CHUNK_SIZE = 1024
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
      "Accept": "audio/mpeg",
      "Content-Type": "application/json",
      "xi-api-key": "3a3173ef3c657dc9add22b3a77b3708f"
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

def main():
    st.header("Chat with PDF 💬")


    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    # st.write(pdf)
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

            # # embeddings
            store_name = pdf.name[:-4]
            st.write(f'{store_name}')
            # st.write(chunks)

            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
                # st.write('Embeddings Loaded from the Disk')s
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)

        if query:
            with st.spinner('Obteniendo la respuesta...'):
                docs = VectorStore.similarity_search(query=query, k=3)

                llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo')
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query+ " Responde en español")
                    print(cb)
                # Convert the response to audio
                audio_file = get_voice_audio(response)
                st.audio(audio_file)
                st.write(response)
            

if __name__ == '__main__':
    main()