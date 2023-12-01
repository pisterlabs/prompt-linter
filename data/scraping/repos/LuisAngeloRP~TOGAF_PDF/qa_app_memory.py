import os
import PyPDF2
import random
import itertools
import streamlit as st
from io import StringIO
from langchain.vectorstores import FAISS
from langchain.chains import (
    RetrievalQA,
    ConversationalRetrievalChain,
    ConversationChain,
)
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SVMRetriever
from langchain.chains import QAGenerationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import CallbackManager
from langchain.chains.conversation.memory import ConversationSummaryMemory


st.set_page_config(page_title="TOGAF con PDFs", page_icon=':book:')

@st.cache_data
def load_docs(files):
    st.info("`Leyendo TOGAF ...`")
    all_text = ""
    pdf_reader = PyPDF2.PdfReader("togaf.pdf")
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    all_text += text
    st.info("`Leyendo documento ...`")
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Por favor, proporcione un archivo txt o pdf.', icon="⚠️")
    return all_text

@st.cache_resource
def process_summary_memory_query(_llm, query: str, summary_memory):
    conversation = ConversationChain(llm=_llm, memory=summary_memory, verbose=True)
    result = conversation.predict(input=query)
    return result

def process_qa_query(query: str, llm, retriever):
    qa = RetrievalQA.from_chain_type(llm= llm, retriever=retriever, chain_type="stuff", verbose=True)
    return qa.run(query)

def create_retriever(_embeddings, splits, retriever_type):
    if retriever_type == "BÚSQUEDA DE SIMILITUD":
        try:
            vectorstore = FAISS.from_texts(splits, _embeddings)
        except (IndexError, ValueError) as e:
            st.error(f"Error al crear el vectorstore: {e}")
            return
        retriever = vectorstore.as_retriever(k=5)

    return retriever

@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):
    st.info("`Dividiendo documento ...`")
    split_method = "RecursiveTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Error al dividir el documento")
        st.stop()

    return splits

def main():
    foot = f"""
    <div style="
        position: fixed;
        bottom: 0;
        left: 30%;
        right: 0;
        width: 50%;
        padding: 0px 0px;
        text-align: center;
    ">
    """
    st.markdown(foot, unsafe_allow_html=True)

    # Agregar CSS personalizado
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .css-card {
            border-radius: 0px;
            padding: 30px 10px 10px 10px;
            background-color: #f8f9fa;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 10px;
            font-family: "IBM Plex Sans", sans-serif;
        }
        .card-tag {
            border-radius: 0px;
            padding: 1px 5px 1px 5px;
            margin-bottom: 10px;
            position: absolute;
            left: 0px;
            top: 0px;
            font-size: 0.6rem;
            font-family: "IBM Plex Sans", sans-serif;
            color: white;
            background-color: green;
        }
        .css-zt5igj {left:0;}
        span.css-10trblm {margin-left:0;}
        div.css-1kyxreq {margin-top: -40px;}
        </style>
        """,
        unsafe_allow_html=True,
    )   

    st.write(
        f"""
        <div style="display: flex; align-items: center; margin-left: 0;">
            <h1 style="display: inline-block;">TOGAF PDF</h1>
            <sup style="margin-left:5px;font-size:small; color: green;">beta v0.1</sup>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Menú")

    embedding_option = st.sidebar.radio(
        "Elige Embeddings", ["OpenAI Embeddings"])

    retriever_type = st.sidebar.selectbox(
        "Elige Retriever", ["BÚSQUEDA DE SIMILITUD"])
    
    chat_type = st.sidebar.selectbox(
        "Elige Retriever", ["Preguntas y Respuestas", "Chat con memoria"])
    
    # Usar RecursiveCharacterTextSplitter como el divisor de texto predeterminado y único
    splitter_type = "RecursiveCharacterTextSplitter"

    if 'openai_api_key' not in st.session_state:
        openai_api_key = st.text_input(
            'Por favor, ingresa tu clave de API de OpenAI o [visita aquí](https://platform.openai.com/account/api-keys)',
            value="", placeholder="Ingresa la clave de API de OpenAI que comienza con sk-")
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            return
    else:
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

    uploaded_files = st.file_uploader("Sube un documento PDF o TXT", type=[
                                      "pdf", "txt"], accept_multiple_files=True)

    if uploaded_files:
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files

        loaded_text = load_docs(uploaded_files)
        st.write("Documentos cargados y procesados.")

        splits = split_texts(loaded_text, chunk_size=1300,
                             overlap=0, split_method=splitter_type)

        num_chunks = len(splits)
        st.write(f"Número de chunks: {num_chunks}")

        if embedding_option == "OpenAI Embeddings":
            embeddings = OpenAIEmbeddings()

        retriever = create_retriever(embeddings, splits, retriever_type)
        
        
        callback_handler = StreamingStdOutCallbackHandler()
        callback_manager = CallbackManager([callback_handler])
        
        llm = ChatOpenAI(
            streaming=True, callback_manager=callback_manager, verbose=True, temperature=0)
        
        summary_memory = None
        summary_memory = ConversationSummaryMemory(llm=llm)

        st.write("Listo para responder preguntas.")

        user_question = st.text_input("Ingresa tu pregunta:")
        if user_question:
            if chat_type == "Preguntas y Respuestas":
                answer = process_qa_query(user_question, llm, retriever)
                st.write("Respuesta:", answer)
            elif chat_type == "Chat con memoria":
                answer = process_summary_memory_query(_llm=llm, query=user_question, summary_memory=summary_memory)


if __name__ == "__main__":
    main()
