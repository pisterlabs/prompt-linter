# Import os to set API key
import os
import langchain
import PyPDF2
import io
import openai
# Import OpenAI as main LLM service
from langchain.llms import OpenAI 
from langchain.embeddings import OpenAIEmbeddings
# Bring in streamlit for UI/app interface
import streamlit as st
from langchain.memory import ConversationBufferMemory

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Pinecone, Chroma

from langchain.chains import ConversationalRetrievalChain

from langchain.chains import ConversationChain
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.
# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
import tempfile
from PyPDF2 import PdfWriter
from langchain.document_loaders import TextLoader
import pinecone
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate, OpenAI, LLMChain

index_name = "langchain-demo"

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
st.session_state.embeddings = OpenAIEmbeddings()

def create_empty_pdf():
    pdf_writer = PdfWriter()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp:
        pdf_writer.write(temp)
        temp_path = temp.name

    return temp_path

def load_data(uploaded_files):
    # Initialize an empty Chroma store
    cnt = 0
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        # now temp_file_path is the path of the temporary file, which you can pass to PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load_and_split()
        # Add documents to the existing Chroma store
        if(cnt==0):
            store = Chroma.from_documents(docs, st.session_state.embeddings, index_name=index_name)
        else:
            store.add_documents(docs)
        cnt+=1

    st.session_state.store = store

    # the rest of your function...

    
    st.session_state.store = store
    if 'past_queries' not in st.session_state:
        st.session_state.past_queries = []
    if 'past_answers' not in st.session_state:
        st.session_state.past_answers = []
    # Add the toolkit to an end-to-end LC
        
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(model_name ="gpt-3.5-turbo",temperature=0), store.as_retriever(), memory=memory,verbose = True)
    return qa
# If the user hits enter


def main(): 
    st.title('🦜🔗 Langchain Document Analytics')
    prompt = st.text_input('Input your prompt here')
    if 'past_queries' not in st.session_state:
        st.session_state.past_queries = []
    if 'past_answers' not in st.session_state:
        st.session_state.past_answers = []

    llm = OpenAI(model_name ="gpt-3.5-turbo",temperature=0)
    qa = ConversationChain(
        llm=llm,
        verbose = True,
        memory=ConversationBufferMemory()
    )
    file_uploaded = False
    
    file_upload = st.file_uploader("Please upload a .pdf file!", type="pdf", accept_multiple_files=True, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    if(len(file_upload)):
        file_uploaded = True
        qa = load_data(file_upload)
    chat_history = []
    if prompt:
        response=""
        if(file_uploaded):
            res = qa({"question":prompt,"chat_history":chat_history})
            response = res["answer"]
            with st.expander('Document Similarity Search'):
                # Find the relevant pages
                search = st.session_state.store.similarity_search_with_score(prompt) 
                # Write out the first 
                st.write(search[0][0].page_content) 
        else:
            response = qa.predict(input = prompt)
        chat_history.append(response)
        st.write(response)
        st.session_state.past_queries.append(prompt)
        st.session_state.past_answers.append(response)
    if len(st.session_state.past_queries) > 0:
        st.subheader('Past Queries and Answers')
        for i, (query, answer) in enumerate(zip(st.session_state.past_queries, st.session_state.past_answers)):
            st.write(f'**Query {i+1}:** {query}')
            st.write(f'**Answer {i+1}:** {answer}')
            st.write('---')

if __name__ == "__main__":
    main()

