import streamlit as st
from dotenv import load_dotenv
import os


@st.cache_resource
def get_api_key():
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    return key


os.environ['OPENAI_API_KEY'] = get_api_key()


@st.cache_resource
def get_chroma_db():
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    persist_directory = "./storage"
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=persist_directory,
                embedding_function=embeddings)
    return db


def talk_to_gpt(query):
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import RetrievalQA

    retriever = get_chroma_db().as_retriever(search_kwargs={"k": 20})
    llm = ChatOpenAI(model_name='gpt-3.5-turbo')

    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever)

    query = f"###Prompt {query}"

    llm_response = qa(query)
    return llm_response["result"]


def main():

    st.title('Titan Emails AI Q&A')

    user_input = st.text_input("You: ")
    if user_input:
        st.write(talk_to_gpt(user_input))
        pass


if __name__ == '__main__':
    main()
