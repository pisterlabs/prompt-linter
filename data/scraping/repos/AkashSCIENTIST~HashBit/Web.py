import streamlit as st
import csv
import os
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import Document
from dotenv import load_dotenv
from log4python.Log4python import log
TestLog = log("ChatLog")
load_dotenv()


def read_csv_into_vector_document(file, text_cols):
    with open(file, newline='\n') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        text_data = []
        for row in csv_reader:
            try:
                text = ' '.join([row[col] for col in text_cols])
                text_data.append(text)
            except:
                pass
        return [Document(page_content=text) for text in text_data]


# Define the layout using neomorphic design
st.set_page_config(
    page_title="HashBit Incident Resolver",
    page_icon=":clipboard:",
)

st.markdown(
    """
    <style>
    .neumorphic {
        padding: 15px;
        background: #272829;
        border-radius: 14px;
        row-gap : 6px;
        
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Top panel
st.title("HashBit PSGCT Incident Resolver")


api_key = os.environ.get("OPENAI_API_KEY")
key_status = True
if not api_key:
    print('OpenAI API key not found in environment variables.')
    key_status = False

data = read_csv_into_vector_document("newdata.csv", ["type", "issue", "resolution", "description"])
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectors = FAISS.from_documents(data, embeddings)
chain = ConversationalRetrievalChain.from_llm(
    llm = ChatOpenAI(
        temperature=0.0, 
        model_name='gpt-3.5-turbo', 
        openai_api_key=api_key
        ), 
    retriever=vectors.as_retriever())

if "messages" not in st.session_state:
    st.session_state.messages = []


if key_status:

    if prompt := st.chat_input("Any Issues ?"):

        query = prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        TestLog.info(query)

        with st.spinner('Waiting for Response'):
            res = chain({"question": query, "chat_history": []})
            TestLog.info(res["answer"])
            st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
            # print(st.session_state['messages'])

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

else:
    st.header("Key not found in .env")
