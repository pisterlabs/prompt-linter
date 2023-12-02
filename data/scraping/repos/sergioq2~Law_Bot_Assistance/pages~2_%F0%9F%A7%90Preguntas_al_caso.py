import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pages.front.htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
import pandas as pd
from PIL import Image, ImageDraw
import boto3
import os

load_dotenv()

aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_access_secret_key = os.getenv("AWS_ACCESS_SECRET_KEY")

client = boto3.client('textract', region_name='us-east-1', aws_access_key_id = aws_access_key, 
                      aws_secret_access_key = aws_access_secret_key)

def read_file(files):
    text = ""
    for uploaded_file in files:
        file_name = uploaded_file.name
        if file_name.lower().endswith('.pdf'):
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            image_bytes = uploaded_file.read()
            response = client.detect_document_text(Document={'Bytes': image_bytes})
            for item in response['Blocks']:
                if item["BlockType"] == "LINE":
                    text = text + item["Text"]
    return text 


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def generate_summary(vectorstore):
    llm = OpenAI(temperature=0)
    chain_1 = load_qa_chain(llm = llm, chain_type='stuff')
    query = 'Haz un resumen del documento de 1 parráfo'
    docs = vectorstore.similarity_search(query)
    summary = chain_1.run(input_documents=docs, question=query)
    return summary

def dataframe_output(output):
    try:
        entries = [entry.strip() for entry in output.split(", ")]

        data = {
            'Name': [entry.split(": ")[0] for entry in entries],
            'Description': [entry.split(": ")[1] for entry in entries]
        }

        df = pd.DataFrame(data)
    except:
        entries = [entry.strip() for entry in output.split(";")]
        data = {
            'Law': [entry for entry in entries]
        }

        df = pd.DataFrame(data)
    return df

def generate_nombres(vectorstore):
    chain_df = load_qa_chain(OpenAI(), chain_type="stuff")
    query = """Identifica todos los nombres propios de personas que están en el documento con su respectivo papel o rol en el documento.
            Ejemplo:
            Sergio : Demandante,Maria : Testigo,Jose : Demandado"""
    docs = vectorstore.similarity_search(query)
    salida = chain_df.run(input_documents=docs, question=query)
    df = dataframe_output(salida)
    return df

def generate_lugares(vectorstore):
    chain_df = load_qa_chain(OpenAI(), chain_type="stuff")
    query = """Identifica todos los nombres de lugares que están en el documento con su respectiva descripción.
            Ejemplo:
            Barcelona : Vereda,Paraiso : Finca,Santuario : Pueblo"""
    docs = vectorstore.similarity_search(query)
    salida = chain_df.run(input_documents=docs, question=query)
    df = dataframe_output(salida)
    return df

def generate_montos(vectorstore):
    chain_df = load_qa_chain(OpenAI(), chain_type="stuff")
    query = """Identifica todos los montos de dinero que están en el documento con su respectiva descripción.
            Ejemplo:
            100 : Valor terreno,200 : Valor casa,300 : Valor demandado,500: Valor escrituras"""
    docs = vectorstore.similarity_search(query)
    salida = chain_df.run(input_documents=docs, question=query)
    df = dataframe_output(salida)
    return df

def generate_leyes(vectorstore):
    chain_df = load_qa_chain(OpenAI(), chain_type="stuff")
    query = """Identifica todas las leyes y artículos que están en el documento con su respectiva descripción.
            Ejemplo:
            Ley 100 : Ley de salud,Artículo 1 : Derecho a la vida,Artículo 2 : Derecho a la salud"""
    docs = vectorstore.similarity_search(query)
    salida = chain_df.run(input_documents=docs, question=query)
    df = dataframe_output(salida)
    return df

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Preguntale al asistente virtual Juris Bot",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Preguntale al asistente virtual Juris Bot :books:")
    user_question = st.text_input("Pregunta cualquier dato sobre el caso o sobre cualquier ley Colombiana:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Tus casos")
        pdf_docs = st.file_uploader(
            "Carga acá todos los documentos del caso y da click en Procesar'", accept_multiple_files=True)

        if "summary" in st.session_state:
            st.subheader("Resumen del caso")
            st.write(st.session_state.summary)
        
        st.sidebar.title("Entidades")
        entidades_options = ["Nombres", "Lugares", "Montos", "Leyes"]
        entidades_selected = st.sidebar.selectbox("Seleccione una opción", entidades_options)


        if pdf_docs and entidades_selected:
            with st.spinner("Procesando..."):
                raw_text = read_file(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                summary = generate_summary(vectorstore)
                st.session_state.summary = summary

                if entidades_selected == "Nombres":
                    nombres_df = generate_nombres(vectorstore)
                    st.subheader("Nombres de personas y su respectivo papel o rol")
                    st.dataframe(nombres_df, height=500)

                elif entidades_selected == "Lugares":
                    lugares_df = generate_lugares(vectorstore)
                    st.subheader("Lugares y su respectiva descripción")
                    st.dataframe(lugares_df, height=500)

                elif entidades_selected == "Montos":
                    montos_df = generate_montos(vectorstore)
                    st.subheader("Montos de dinero y su respectiva descripción")
                    st.dataframe(montos_df, height=500)

                elif entidades_selected == "Leyes":
                    leyes_df = generate_leyes(vectorstore)
                    st.subheader("Leyes y sus respectivos artículos")
                    st.dataframe(leyes_df, height=500)

                st.session_state.conversation = get_conversation_chain(vectorstore)
              
if __name__ == '__main__':
    main()