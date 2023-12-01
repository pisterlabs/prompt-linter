from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain.chat_models import ChatOpenAI

chat_model = ChatOpenAI()

st.title('에세이 작성')

content = st.text_input('주제를 정해주세요.')

if st.button('요청하기'):
    with st.spinner('waiting...'):
        result = chat_model.predict(content+"에대한 essay를 모두에게 공감이가고 낭만적으로 에세이 문장구조에 맞게 작성하고줘")
                                               
        st.write(result)