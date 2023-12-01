import json
import pandas as pd
import streamlit as st
from pathlib import Path
import core.templates as pr
from langchain import OpenAI, SQLDatabase
from langchain import PromptTemplate, LLMChain
import re
import core.interface as it
import os
import html
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document
import logging
from langchain.vectorstores import PGEmbedding
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

st.set_page_config(
    page_title="Boss Services APP",
    page_icon="👋",

)



#connection string "postgresql+psycopg2://harrisonchase@localhost:5432/test3"
CONNECTION_STRING = st.secrets['CONNECTION_STRING']
#collection name
COLLECTION_NAME = st.secrets['COLLECTION_NAME']

os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']

#EMBEDING
embeddings = OpenAIEmbeddings()

llm = ChatOpenAI(temperature = 0.0, model = 'gpt-3.5-turbo-16k', verbose=True)
#llm = OpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, verbose=True)

#Store
store = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)

retriever = store.as_retriever()

st.title("👨‍💻 Asistente Empresarial")
st.write (":warning: esto es una versión preliminar a falta de refinamientos.")


PROMPT = pr.prompts_default()

question = st.text_area("Inserta tu consulta")
boton = st.button('Enviar')
if question and  boton:
   
    with st.spinner('Cargando...'):
        llm_chain =   RetrievalQA.from_chain_type(
                          llm=llm,   
                          chain_type="stuff", 
                          retriever=retriever,
                          verbose=True,
                      )
        simple_prompt = str(PROMPT) + str(question)
        respuesta =  llm_chain.run(simple_prompt)
        
        st.write("### Respuesta")   
     
          
        # Decode the response.       
        decoded_response = it.decode_response(respuesta)
        # Write the response to the Streamlit app.
        print(decoded_response)
        it.write_response(decoded_response)
        try: 
            st.write("---")
        except  ValueError:
            st.markdown(
                "No se ha podido procesar la pregunta, por favor provea de más contexto para poder procesar la información. Si el problema persiste, contecte con soporte"
            )
            st.markdown(
                "Detalle"      )
            st.markdown(
                ValueError      )