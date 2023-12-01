import os
import pickle
import requests
import tempfile
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext
from PyPDF2 import PdfReader
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

# Sidebar contents
with st.sidebar:
    st.image('your_logo.png', width=300)  # add your logo
    st.title('LLM Chat App 🤗💬')
    st.markdown('''
    **About**

    This app is a LLM-powered chatbot that utilizes:
    * Streamlit for an interactive UI
    * LangChain for text processing
    * OpenAI's LLM model for answering queries

    ''')
    add_vertical_space(5)
    st.markdown('Made with ❤️ by [Prompt Engineer](https://youtube.com/@engineerprompt)')
    add_vertical_space(5)
    st.markdown('Modified with 💪 by [Manrrolo](https://manrrolo.github.io/Page/)')

load_dotenv()

# Replace with your OpenAI API key
os.environ["OPENAI_API_KEY"] = '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'
eleven_api_key = "880ec17adcf3ee790a09c2ef884a6e8c"
selected_voice_id = "2"  # Antoni voice

def main():
    st.header("Chat with PDF 💬")
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
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

        query = st.text_input("Ask questions about your PDF file:")

        if query:
            modelo = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'))
            service_context = ServiceContext.from_defaults(llm_predictor=modelo)
            index = GPTVectorStoreIndex.from_documents(VectorStore, service_context = service_context)
            
            respuesta = index.as_query_engine().query(query + " Responde en español")
            
            st.markdown(f"## Response: 🗨️")
            st.markdown(f"<div style='background-color:lightgrey;padding:10px;border-radius:5px;'>{respuesta.response}</div>", unsafe_allow_html=True)

            url = "https://api.elevenlabs.io/v1/text-to-speech/" + selected_voice_id
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": eleven_api_key
            }
            data = {
                "text": respuesta.response,
                "model_id" : "eleven_multilingual_v1",
                "voice_settings": {
                    "stability": 0.4,
                    "similarity_boost": 1.0
                }
            }

            response = requests.post(url, json=data, headers=headers)
            CHUNK_SIZE = 1024
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                f.flush()
                temp_filename = f.name

            st.markdown(f"## Response in Audio: 🎧")
            st.audio(temp_filename, format='audio/mp3')

        if st.button('Clear All'):
            st.experimental_rerun()

if __name__ == '__main__':
    main()
