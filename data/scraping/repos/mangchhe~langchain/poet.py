from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
import streamlit as st

load_dotenv()
chat_model = ChatOpenAI()

st.title('인공지능 시인')
topic = st.text_input('시의 주제를 제시해주세요.')

if topic:
    with st.spinner('Wait for it...'):
        poet = chat_model.predict(topic + " 주제에 대한 시를 써줘")
    st.write(poet)