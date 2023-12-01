import streamlit as st
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

st.set_page_config(page_title="Bibl-IA", page_icon="images/biblIA.png")
st.image("images/biblIA.png", width=100)
st.title('Bibl-IA')
st.subheader('A sua assistente para ler a Bíblia utilizando Inteligência Artificial')

content = st.text_area('Como está se sentindo?')

def generate_response(input_text):
  llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key, max_tokens=2048)
  st.info(llm(input_text))

with st.form('my_form'):
  text = 'Hoje estou me sentindo: ' + content + '\n'
  text += 'E gostaria de ler a Bíblia sobre: ' + content + '\n'

  submitted = st.form_submit_button('Enviar')
  if submitted:
    generate_response(text)
