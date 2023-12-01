import streamlit as st
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
import base64
import time

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"] 

def set_background(svg_file):
    def get_base64(file_path):
        with open(file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    bin_str = get_base64(svg_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/svg+xml;base64,%s");
        background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar contents
with st.sidebar:
    st.title('DocReader 📖')
    st.markdown('''
    ## About
    Strumento basato sull'AI 🤖che ottimizza l'attività di ricerca in dottrina e giurisprudenza per i professionisti 
    del diritto ⚖️
 
    ''')
    st.image('./assett/DocReaderLOGO.png', width=200)

    add_vertical_space(5)
    st.write('Made by [Fiacconi Industries](https://fiacconindustries.netlify.app/) 🚀')
 

def main():
    st.header("DocReader 📖")
    set_background('./assett/background3.svg')
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
 
    # st.write(pdf)
    if pdf is not None:
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
 
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
 
        if query:
            progress_text = ("We are searching for your answer 🔍")
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1, text=progress_text)

            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
 
if __name__ == '__main__':
    main()
