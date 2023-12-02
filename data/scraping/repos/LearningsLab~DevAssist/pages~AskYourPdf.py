import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from services.GetEnvironmentVariables import GetEnvVariables
from services.TextChunkSplitterService import TextChunkSplitterService
from services.CreateEmbeddingService import CreateEmbeddingService

st.set_page_config(page_title="Ask your PDF")
st.header("Ask your PDF 💬")

# get env variables 
env_vars = GetEnvVariables()
OPENAI_API_KEY = env_vars.get_env_variable('openai_key')

# upload file
pdf = st.file_uploader("Upload your PDF", type="pdf")

# extract the text
if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # split into chunks
    splitters = TextChunkSplitterService('/n',250,200,len)
    chunks = splitters.split_text(text)
    st.write(chunks)

    # create embeddings
    embeddingsIni = CreateEmbeddingService()
    embeddings = embeddingsIni.create_embeddings('OpenEmbeddings')
    
    # create indexing
    knowledge_base = FAISS.from_texts(chunks, embeddings)
        
    user_question = st.text_input("Ask a question about your PDF:")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
        llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          st.write(cb)
           
        st.write(response)
