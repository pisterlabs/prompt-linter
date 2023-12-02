import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain import OpenAI, PromptTemplate
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

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

def generate_correcction(vectorstore):
    llm = OpenAI(temperature=0)
    chain_1 = load_qa_chain(llm = llm, chain_type='stuff')
    query = '''Simula que eres un abogado profesional y trata de conservar el lenguaje propio
    del campo legal. Uaz una corrección al texto tratand siguiendo los protocols legales y profesionales'''
    docs = vectorstore.similarity_search(query)
    correction = chain_1.run(input_documents=docs, question=query)
    return correction

def main():
    load_dotenv()
    st.markdown(
        """
        <style>
        .reportview-container .main .block-container{
            max-width: 1000px;
            padding-top: 2rem;
            padding-right: 2rem;
            padding
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Corregir redacción")
    st.markdown("Escribe un texto y el sistema te ayudará a corregir la redacción")
    texto = st.text_area("Escribe o copia un texto", height=300)

    if st.button("Corregir redacción"):
        text_chunks = get_text_chunks(texto)
        vectorstore = get_vectorstore(text_chunks)
        correction = generate_correcction(vectorstore)
        st.write(correction)

if __name__ == "__main__":
    main()