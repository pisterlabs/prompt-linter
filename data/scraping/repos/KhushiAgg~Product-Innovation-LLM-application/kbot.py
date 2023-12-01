import os
import pickle
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter #for splitting text
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS  #facebook AI Similarity Search for vectorization
from langchain import HuggingFaceHub #for loading LLM
from langchain.chains.question_answering import load_qa_chain #for setting up QnA chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pprint import pprint


# Sidebar
with st.sidebar:
    st.title("K LLM Chatbot")
    st.markdown('''
    ## About
    This is a LLM-powered customer support chatbot build using:
    - [Streamlit](https://streamlit.io/)
    - [Langchain](https://www.langchain.com/)
    - [Huggingface](https://huggingface.co/) LLM Model

    ''')
    add_vertical_space(5)
    st.write('Made by [Khushi Agarwal](https://github.com/KhushiAgg)')


def main():
    st.header("🫂KK Customer Support Bot")

    load_dotenv()

    #Upload your pdf file
    pdf = st.file_uploader("Upload your pdf", type='pdf')
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text to chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len)
        chunks = text_splitter.split_text(text=text)
        # st.write(chunks)
        
        file_name = pdf.name[:-4]

        # to save incurring cost for creating vector dbs again and again 
        # we execute a simple if-else for checking if v db exists
        if os.path.exists(f"{file_name}.pkl"):
            with open(f"{file_name}.pkl", "rb") as f:
                faiss_index = pickle.load(f)
            # st.write("Embeddings loaded from the Disk")
        else:
            #Initialize embeddings
            embeddings = HuggingFaceEmbeddings()
            # PDF chunks --> Embeddings --> Store vectors
            faiss_index = FAISS.from_texts(chunks, embeddings)

            with open(f"{file_name}.pkl", "wb") as f:
                pickle.dump(faiss_index, f)
            
            # st.write("Embeddings created")
        
        # Query from your pdf
        query = st.text_input("Ask questions about your PDF file: ")
        # st.write(query)
        if query:
            docs = faiss_index.similarity_search(query = query)
            
            llm=HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", model_kwargs={"temperature":0.1, "max_length":512})
            
            # Setting up a similarity search i.e. query most similar to the chunk is retrieved and answeres.
            chain = load_qa_chain(llm = llm, chain_type = "stuff")
            response = chain.run(input_documents=docs, question=query)
            # st.write(response)
            
            # Setting up ConversationalRetrievalChain this chain builds on RetrievalQAChain to provide a chat history component.
            # Setting up the retrieval process
            retriever = faiss_index.as_retriever()
            # Creating memory of conversation
            memory = ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True,
                output_key='answer')
            # Set up Consersation chain
            chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
            )
            
            # Display answers
            result = chain({"question": query})
            st.write(result["answer"])
            pprint(memory.buffer)

        


if __name__ == '__main__':
    main()