#Importing dependencies
import os 
import langchain
import streamlit as st 
import time
import re

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessage
)

from langchain.chains import LLMChain

apikey = os.getenv('OPENAI_API_KEY')

with open('hobbies.csv', 'r') as f:
    hobbies_options = f.read().splitlines()
with open('traits.csv', 'r') as f:
    traits_options = f.read().splitlines()

#App framework
st.title('Coolblue Sinterklaas gedichten ✍️')

st.markdown("""
    Welkom bij de Coolblue Sinterklaas gedichten generator!

    """)

name = st.text_input('Voor wie is dit cadeau?')
gender = st.radio('Selecteer zijn/haar gender:', ['Vrouw', 'Man'])
hobby = st.multiselect('Wat zijn zijn/haar hobby\'s? (selecteer er 2)', hobbies_options, max_selections=2)
traits = st.multiselect('Wat zijn zijn/haar goede eigenschappen? (selecteer er 2)',traits_options,max_selections=2)
product_type_name = st.text_input('Welk cadeau heb je gekocht voor hem/haar?')
product = st.text_area('Vul hier de product informatie in')

#Chatmodel 

chat_model= ChatOpenAI(temperature=0.6, model="gpt-4")

#Prompt template

system_message_prompt = SystemMessagePromptTemplate.from_template("""Je schrijft Sinterklaasgedichten voor de klanten van Coolblue.

Schrijf de gedichten op basis van informatie over de klant en het product dat ze hebben gekocht.

Het gedicht moet grappig, positief en blij. Verklap het product niet maar draai er omheen.

Gebruik maximaal 8 regels.
""")
human_message_prompt = HumanMessagePromptTemplate.from_template("""Informatie over de klant:
- Naam: {name}
- Voornaamwoorden: {pronouns}
- Hobbies: {hobby}
- Goede eigenschappen: {traits}

Informatie over het product:
- {product_type_name}
{product}
""")
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

#LLM Chain

gedicht_chain = LLMChain(llm=chat_model, prompt=chat_prompt, verbose = True)

#show stuff
if st.button('Vraag G-Piet-R om een gedicht!'):
    try:
        if object:
            response = gedicht_chain.run({
                "name": name,
                "pronouns": 'Zij/haar' if gender == 'Vrouw' else 'Hij/hem', 
                "hobby": ','.join(hobby),
                "traits": ','.join(traits), 
                "product_type_name": product_type_name,
                "product": product,
            })
            st.text(response)
    except Exception as e:
        st.error(f"an error occurred:{e}")




