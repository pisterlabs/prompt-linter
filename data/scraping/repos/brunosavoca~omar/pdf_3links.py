import streamlit as st
import openai
from PyPDF2 import PdfReader 
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain 

# Esto es para el sidebar, donde el usuario ingresa la clave de la API de OpenAI
openai_api_key = st.sidebar.text_input("Introduce tu OpenAI API Key")

# Interacción con PDFs
st.header('Interactuar con PDFs')

pdf_file = st.file_uploader("Cargá tu PDF", type=["pdf"])

if pdf_file is not None:
    # Process the uploaded file
    reader = PdfReader(pdf_file)

    # read data from the file and put them into a variable called raw_text
    raw_text =''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    docsearch = FAISS.from_texts(texts, embeddings)

    #acá se puede utilizar diferentes modelos, en este caso será OpenAI
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    query = st.text_input("Qué necesitas saber de este documento?", "breve resumen de este documento")
    if query:
        docs = docsearch.similarity_search(query)
        answer = chain.run(input_documents=docs, question=query)
        st.write(answer)

    # Añadiendo los botones
    if st.button("Obtener un resumen del PDF"):
        summarized_text = summarize(raw_text)
        st.text_area("Resumen del PDF", summarized_text)

    if st.button("Obtener una lista de 5 puntos del documento"):
        points = summarize(raw_text, split=True)[:5]
        for i, point in enumerate(points, start=1):
            st.write(f"{i}. {point}")
