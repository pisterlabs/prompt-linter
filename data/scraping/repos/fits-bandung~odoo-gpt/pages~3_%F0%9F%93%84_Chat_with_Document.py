import streamlit as st  
import utils.login as login
import utils.logout as logout
import time

from utils.database import User, Message, app, db_sqlalchemy, inspect_db
from utils import sidebar as sidebar
import pandas as pd

import json
from PyPDF2 import PdfReader
import pickle

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI

from langchain.chat_models import ChatOpenAI

from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from dotenv import load_dotenv
import base64



os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '.credentials/erp-2017-d844ca1fcc74.json'
load_dotenv('.credentials/.env')
MODEL = 'gpt-3.5-turbo'
API_O = os.environ['OPENAI_KEY']






def run():
    sidebar.run()
    mobile_phone =  st.session_state["mobile_phone"]
    
    st.markdown("""
        # Chat with PDF Document
    """)



    #Mempersiapkan folder untuk menyimpan file Document
    PKL_FOLDER = "static/documents/pkl"
    if not os.path.exists(PKL_FOLDER):
        os.makedirs(PKL_FOLDER)
    

    # Mengunggah file PDF
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    #reset session state untuk message
    if "message" not in st.session_state:
        st.session_state["message"] = ""


     
    if pdf is not None:
        
        #total cost
        if "total_cost" not in st.session_state:
            st.session_state["total_cost"] = 0


        #buat button untuk download file pdf
        st.download_button(f"Download : {pdf.name}", "pdf", f"{pdf.name}", "application/pdf", key=None, help=None)
        # st.write(type(pdf))
        pdf_reader = PdfReader(pdf)
        text = "".join(page.extract_text() for page in pdf_reader.pages)

        # Memecah teks menjadi potongan-potongan yang dapat dikelola
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text=text)


        with st.expander(f"**{pdf.name}** : {len(pdf_reader.pages)} halaman"):
            st.write(chunks)

        # if st.button("Load Vector Store"):



        # Menyimpan nama file PDF
        store_name = pdf.name[:-4]
        pkl_path = os.path.join(PKL_FOLDER, f"{store_name}.pkl")
        print(f'Pkl Path = {pkl_path}')



         # Memeriksa apakah embeddings sudah ada
        if os.path.exists(pkl_path):
            with open(pkl_path,"rb") as f:
                vectorstore = pickle.load(f)
                print(f'Vector Strore is Exist = {str(vectorstore)}')

        else:
            print(f'Get Embedding from OpenAI({len(chunks)} chunks)')
            embeddings = OpenAIEmbeddings(openai_api_key=API_O)

            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            try:
                with open(pkl_path, "wb") as f:
                    pickle.dump(vectorstore, f)
            except Exception as e:
                st.write(f"Error saving file: {e}")
            

         

        with st.expander(f'File: "**{f.name}**"'):

            st.write(vectorstore)
            st.write("---")
            total_cost = st.empty()

           




        #Definisikan container bernama message_box berupa expander yang empty untuk bisa di isi oleh message dari session state
        message_box = st.expander(f"**Chat History**", expanded=True)

        with message_box:
            spinner = st.empty()
            if st.session_state["message"] != "":
                if st.button("Clear Chat History"):
                    st.session_state["message"] = ""
                    st.session_state["total_cost"] = 0
                    st.experimental_rerun()

        # Menerima pertanyaan dari pengguna
        message = ""
        query = st.chat_input(placeholder="Ajukan pertanyaan tentang terkait file pdf unggahan Anda",)

        if query:
            #tambahkan spinner
            with spinner:
                with st.spinner("Similarity Searching ..."):
                    docs = vectorstore.similarity_search(query=query, k=5)

                # with st.expander(f"**Similarity Search** : {len(docs)} dokumen", expanded=False):
                #     st.write(docs)
          
                with st.spinner("Call LLM ..."):
                    llm = ChatOpenAI(temperature=0, openai_api_key=API_O, model_name=MODEL, verbose=True)
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=query)
                        print(f'Output = {response}')
                        st.session_state["total_cost"] += cb.total_cost
                        callbact_text = f"Total Tokens: {cb.total_tokens}\nPrompt Tokens: {cb.prompt_tokens}\nCompletion Tokens: {cb.completion_tokens}\nCost (IDR): IDR {cb.total_cost*15000}\nTotal Cost (IDR): IDR {st.session_state['total_cost']*15000}    "
                        print(callbact_text)
                           
                    with total_cost:
                        callbact_text = callbact_text.replace("\n", "\n\n")
                        st.write(callbact_text)


            message = f'***Q: {query}*** \n\nA: {response}\n\n'
            st.session_state["message"] += message



            #Tampilkan message dari session state
            message_box.write(st.session_state["message"])






         
       
        



                        # st.success("Records from Table " + str(i+1) + " are deleted")
                        # time.sleep(1) # wait some time then refresh the page
                        # st.experimental_rerun()


if st.session_state.get('token') is None:
    st.error("You are not logged in!")
    login.run()

else:

    #Tampilkan kredensial
    run()
