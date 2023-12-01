import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from PIL import Image

image = Image.open('images/wazo.png')

st.set_page_config(
    page_title="Wazobia (The Nigerian Translator)",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

load_dotenv()

langs = ["Hausa", "Igbo", "Yoruba", "English"]

# Hide Streamlit style
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            <style>
"""

st.markdown(hide_st_style, unsafe_allow_html=True)

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image(image)
st.markdown('# Wazobia (The Nigerian Translator)')

with st.sidebar:
     language = st.radio('Select language to translate to:', langs)

st.markdown('### Wazobia Translate')
prompt = st.text_input('Enter text here')

trans_template = PromptTemplate(
    input_variables=['trans'],
    template='Your task is to translate this text to ' + language + 'TEXT: {trans}'
)

# Memory
memory = ConversationBufferMemory(input_key='trans', memory_key='chat_history')

# LLMs
llm = OpenAI(model_name="text-davinci-003", temperature=0)
trans_chain = LLMChain(llm=llm, prompt=trans_template, verbose=True, output_key='translate', memory=memory)

# If there's a prompt, process it and write out response on screen
if st.button("Translate"):
    if prompt:
        response = trans_chain({'trans': prompt})
        st.info(response['translate'])
