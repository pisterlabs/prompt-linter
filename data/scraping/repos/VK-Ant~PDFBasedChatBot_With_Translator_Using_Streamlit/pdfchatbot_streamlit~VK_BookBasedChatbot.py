import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfFileReader, PdfFileWriter,PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

import pickle
import os
#load api key lib
from dotenv import load_dotenv
import base64


#Background images add function
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpeg"};base64,{encoded_string.decode()});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
#add_bg_from_local('images.jpeg')  

#sidebar contents

with st.sidebar:
    st.title('🦜️🔗VK - PDF BASED LLM-LANGCHAIN CHATBOT🤗')
    st.markdown('''
    ## About APP:

    The app's primary resource is utilised to create::

    - [streamlit](https://streamlit.io/)
    - [Langchain](https://docs.langchain.com/docs/)
    - [OpenAI](https://openai.com/)

    ## About me:

    - [Linkedin](https://www.linkedin.com/in/venkat-vk/)
    
    ''')

    add_vertical_space(4)
    st.write('💡All about pdf based chatbot, created by VK🤗')

load_dotenv()

def main():
    st.header("📄Chat with your pdf file🤗")

    #upload a your pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    st.write(pdf.name)

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text+= page.extract_text()

        #langchain_textspliter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )

        chunks = text_splitter.split_text(text=text)

        
        #store pdf name
        store_name = pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                vectorstore = pickle.load(f)
            #st.write("Already, Embeddings loaded from the your folder (disks)")
        else:
            #embedding (Openai methods) 
            embeddings = OpenAIEmbeddings()

            #Store the chunks part in db (vector)
            vectorstore = FAISS.from_texts(chunks,embedding=embeddings)

            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(vectorstore,f)
            
            #st.write("Embedding computation completed")

        #st.write(chunks)
        
        #Accept user questions/query

        query = st.text_input("Ask questions about related your upload pdf file")
        #st.write(query)

        if query:
            docs = vectorstore.similarity_search(query=query,k=3)
            #st.write(docs)
            
            #openai rank lnv process
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm=llm, chain_type= "stuff")
            
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = query)
                print(cb)
            st.write(response)



if __name__=="__main__":
    main()