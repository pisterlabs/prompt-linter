import streamlit as st
import openai
from utils import getMessage, cekApikey
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("API_KEY")


st.set_page_config(page_title="Pocketmed", page_icon= "💊")
st.sidebar.header("Dokter Virtual")
st.write("### DR. REHAN WANGSAFF (DOKTER VIRTUAL)")
initBot = [{'role': 'user', 'content': '''Disini kamu berperan sebagai seorang dokter bernama Dokter Rehan Wangsaff.
                                  Kamu adalah seorang dokter spesialis kulit yang sangat ahli dan sedang berbicara dengan saya sebagai seorang pasien.
                                  Jawablah pertanyaan-pertanyaanku ini dengan baik dan benar sebagaimana seorang dokter.
            ketika menjawab, jangan gunakan "Dr. Rehan: ", "Dokter Rehan Wangsaff: " dan lain sebagainya, langsung saja jawab jawabanmu'''}]

if "messages" not in st.session_state:
    st.session_state.messages = initBot

if not cekApikey():
    prop = st.text_input("Masukkan apikey openaimu disini untuk berbicara dengan dokter virtual", type="password", placeholder="APIKEY")
    if prop:
        openai.api_key = prop


for i, message in enumerate(st.session_state.messages):
    if i == 0:
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Masukkan pesan"):
    prompt = f"Kamu: {prompt}"
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    response = getMessage(st.session_state.messages)
    
    st.session_state.messages.append({"role" : "assistant", "content" : response})
    with st.chat_message("assistant"):
        st.markdown(f"Dr. Rehan: {response}")