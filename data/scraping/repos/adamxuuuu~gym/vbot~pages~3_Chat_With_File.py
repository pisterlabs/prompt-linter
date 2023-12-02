import streamlit as st
from langchain.llms import OpenAI, LlamaCpp
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from utils.callbacks import StreamHandler
from config import (EMBEDDING, LLM)

def generate_response(uploaded_file, query_text, stream_handler):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        # Select embeddings
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING)
        # Create a vectorstore from documents
        db = FAISS.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 4, 'lambda_mult': 0.25}
        )
        # Create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=LlamaCpp(model_path=LLM, n_ctx=2048, temperature=0.01),
            chain_type='stuff',
            retriever=retriever
        )
        return qa.run(query_text, callbacks=[stream_handler])


# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# File upload
uploaded_file = st.file_uploader('Upload an document (1kb usually takes 120 seconds to process)', type='txt', )
print(uploaded_file)
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not(uploaded_file))

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file))
    if submitted:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                stream_handler = StreamHandler(st.empty())
                response = generate_response(uploaded_file, query_text, stream_handler)
