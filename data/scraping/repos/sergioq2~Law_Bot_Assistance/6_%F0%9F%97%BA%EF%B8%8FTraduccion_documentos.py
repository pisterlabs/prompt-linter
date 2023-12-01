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
import  docx2txt
import openai
from docxtpl import DocxTemplate
import os
import json

docx_tpl = DocxTemplate("./docs/traduccion.docx")

openai_api_key = os.getenv("OPENAI_API_KEY")
aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_access_secret_key = os.getenv("AWS_ACCESS_SECRET_KEY")

client = boto3.client('textract', region_name='us-east-1', aws_access_key_id=aws_access_key,
                       aws_secret_access_key=aws_access_secret_key)

def read_file(files):
    text = ""
    file_name = files.name
    if file_name.lower().endswith('.pdf'):
        pdf_reader = PdfReader(files)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        image_bytes = files.read()
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

def detect_language(vectorstore):
    llm = OpenAI(temperature=0)
    chain_1 = load_qa_chain(llm = llm, chain_type='stuff')
    query = 'Detecta el idioma de este documento y devuelvelo en español en una sola palabra'
    docs = vectorstore.similarity_search(query)
    source_language = chain_1.run(input_documents=docs, question=query)
    return source_language

def translate_text(text, source_language, target_language):
    prompt = f"Translate the following '{source_language}' text to '{target_language}': {text}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates text."},
            {"role": "user", "content": prompt}
        ],
        n=1,
        stop=None,
        temperature=0,
    )

    translation = response.choices[0].message.content.strip()
    return translation


def main():
    load_dotenv()
    st.title("Traducción de texto")
    st.markdown("Sube el documento que deseas traducir")
    file = st.file_uploader("Sube el documento que deseas traducir", type=['pdf', 'docx', 'jpg'])
    if file is not None:
        texto = read_file(file)
        text_chunks = get_text_chunks(texto)
        vectorstore = get_vectorstore(text_chunks)
        language1 = detect_language(vectorstore)
        st.write(f"El idioma detectado es: {language1}")

    language2 = st.selectbox("Escoge el idioma al que deseas traducrlo",
                              ["Español", "Inglés", "Francés", "Alemán", "Italiano", "Portugués", 
                               "Ruso", "Chino", "Japonés", "Coreano", "Árabe", "Hindi", "Bengalí"])
    
    def creacion_documento():
        context = {
            'traduccion': traduccion
            }
        docx_tpl.render(context)
        docx_tpl.save(f'Traducciones/traduccion.docx')


    if st.button("Traducir documento"):
        traduccion = translate_text(texto, language1, language2)
        st.write(traduccion)
        creacion_documento()


if __name__ == "__main__":
    main()