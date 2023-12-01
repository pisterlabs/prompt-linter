import streamlit as st
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

import streamlit as st
from streamlit_chat import message
import pinecone
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI

from PIL import Image

#os.environ['OPENAI_API_KEY'] 
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

index_name = 'research-helper'

def generate_response(query_text, index_name, embeddings):
    
    chat_model = ChatOpenAI()

    papers_db =  Pinecone.from_existing_index(index_name, embeddings)

    # Create retriever interface
    retriever = papers_db.as_retriever()
    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm=chat_model,\
                                      chain_type='stuff',
                                      retriever=retriever)
    
    return qa.run(query_text)


st.set_page_config(page_title="Ask ArXiv", page_icon=":computer:", layout="wide")

app_title = '<p style="font-family:sans-serif; color:White; \
    text-align: center; font-size: 62px;">🧑‍🔬Ask ArXiv🧑‍💻</p>'

st.markdown(app_title, unsafe_allow_html=True)

#opening the image

# with st.sidebar.container():
#     image = Image.open("arxiv.jpeg")
#     # image = image.resize((100, 100))
#     st.image(image, use_column_width=True)
#     # st.sidebar.image(image)


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

query_text = st.text_input("You: ", "Hello!", label_visibility="visible", key="input")

# Form input and query

response = generate_response(query_text=query_text,
                             index_name='research-helper', \
                             embeddings=embeddings)

st.session_state.past.append(query_text)
st.session_state.generated.append(response)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):

        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user", \
                avatar_style="lorelei", seed=123)
        message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", \
                seed=123)
