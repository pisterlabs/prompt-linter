# Bring in deps
import os
import pickle
from apikey import apikey
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS



# App Framework

# Sidebar
with st.sidebar:

    st.title('MediMitra')
    st.markdown('''
    ## About 
        MediMitra is a chatbot which 
        can help in providing 
        cures and suggesting 
        you a proper 
        treatment.
            ''')
    add_vertical_space(5)
    st.write('Made by TechSouls')


def main():
    os.environ['OPENAI_API_KEY'] = apikey
    st.image("./Logo.png")
    with st.container():

        st.write("🤖")
        st.text(""" 
            Hey!👋  I am MediMitra. Unlike your other mitras
            I will stand by your side in bad times.
            """)

    #pdf file
    pdf_reader = PdfReader("./Dataset.pdf")
    text = ""
    for page in pdf_reader.pages:
        text+=page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)

    #embeddings
    store_name = "Dataset"
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl","rb") as f:
            VectorStore = pickle.load(f)
        #st.write("Embeddings loaded from the disk")
    else:
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
        with open(f"{store_name}.pkl","wb") as f:
            pickle.dump(VectorStore,f)
        
        #st.write("Embeddings Computation Completed")

    #Accept user questions
    query = st.text_input("Prompt me your symptoms")
    intensity = st.slider("Scale your symptoms",1,10,1)

    if query:
        docs = VectorStore.similarity_search(query=query, k=11-intensity)

        llm = OpenAI(temperature=0)
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        st.write(response)

        #st.write(docs)


if __name__ == '__main__':
    main()



