import secret_keys
import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

openai_api_key = os.environ.get('OPENAI_API_KEY')

def generate_chain(uploaded_file):
    # Load document if file is uploaded
    if uploaded_file is not None:

        print("indexing")
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(OpenAI(openai_api_key=openai_api_key, temperature=0), verbose = True, retriever=retriever, memory=memory)
        
        return qa

# Page title
st.set_page_config(page_title='🦜🔗 Ask your Document')
st.title('🦜🔗 Ask your Document')


# File upload
uploaded_file = st.file_uploader('Upload an article', type='txt')
with st.form('index'):
    submitted = st.form_submit_button('Index', disabled=not(uploaded_file))
    if submitted:
        with st.spinner('Calculating...'):
            
            if "qa_chain" not in st.session_state:
                st.session_state["qa_chain"] = generate_chain(uploaded_file)
                

if "qa_chain" in st.session_state:
    qa = st.session_state["qa_chain"]

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = qa.run(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
