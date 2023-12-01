from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma


if "text_splitter" not in st.session_state:
    pass

if "embeddings" not in st.session_state:
    pass

if "chat" not in st.session_state:
    pass

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    # Lo hacemos para tener una continuidad entre los fragmentos
    chunk_overlap=200,
    length_function = len
)
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
chat = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    temperature=0.0
)


def query_function(ordenanza, query):
    ml_papers = []
    loader = TextLoader(f"./ordenanzas_txt/ORD_{ordenanza}.txt")
    data = loader.load()
    ml_papers.extend(data)
    documents = text_splitter.split_documents(ml_papers)

    if "currently_ord" not in st.session_state or st.session_state.currently_ord != ordenanza:
        st.session_state.currently_ord = ordenanza
        st.session_state.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
        )
    
    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k":1}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type='stuff',
        retriever=retriever
    )

    return qa_chain.run(query)

