# Importing dependencies
import os 
import langchain
import streamlit as st 
import time
import re
import requests
from io import BytesIO
import traceback
from elevenlabs import clone, generate, play, set_api_key
from elevenlabs.api import History

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessage
)

from langchain.chains import LLMChain

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "sinterklaas-finetuning"

# Set the Eleven Labs API Key (replace with your key)
set_api_key(os.environ.get("ELEVEN_LABS_API_KEY"))

apikey = os.getenv('OPENAI_API_KEY')

with open('hobbies.csv', 'r') as f:
    hobbies_options = f.read().splitlines()
with open('badtraits.csv', 'r') as f:
    traits_options = f.read().splitlines()

#App framework
st.title('Coolblue Sinterklaas gedichten ✍️')

st.markdown("""
    Welkom bij de Coolblue Sinterklaas gedichten generator!

    """)

name = st.text_input('Voor wie is dit cadeau?')
hobby = st.multiselect('Wat zijn zijn/haar hobby\'s? (selecteer er 2)', hobbies_options, max_selections=2)
traits = st.multiselect('Wat zijn zijn/haar slechte eigenschappen? (selecteer er 2)',traits_options,max_selections=2)
product_type_name = st.text_input('Welk cadeau heb je gekocht voor hem/haar?')
product = st.text_area('Vul hier de product informatie in')

#Chatmodel 

chat_model= ChatOpenAI(temperature=0.6, model="gpt-4")

#Prompt template

system_message_prompt = SystemMessagePromptTemplate.from_template("""Je schrijft Sinterklaasgedichten voor de klanten van Coolblue.

Schrijf de gedichten op basis van informatie over de klant en het product dat ze hebben gekocht.

Het gedicht moet grappig, positief en blij. Verklap het product niet maar draai er omheen.

Gebruik maximaal 8 regels.

Antwoord met "Jij gaat mee in de zak naar Spanje" wanneer iemand een naam ingeeft die beledigend is.
""")
human_message_prompt = HumanMessagePromptTemplate.from_template("""Informatie over de klant:
- Naam: {name}
- Hobbies: {hobby}
- Slechte eigenschappen: {traits}

Informatie over het product:
- {product_type_name}
{product}
""")
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

#LLM Chain

gedicht_chain = LLMChain(llm=chat_model, prompt=chat_prompt, verbose = True)


# show stuff
if st.button('Vraag G-Piet-R op een gedicht!'):
    try:
        if object:
            response = gedicht_chain.run({
                "name": name,
                "hobby": ','.join(hobby),
                "traits": ','.join(traits), 
                "product_type_name": product_type_name,
                "product": product,
            })
            st.text(response)

            # Generate audio from text using Eleven Labs
            model_id= "eleven_multilingual_v2"
            voice_id = os.environ.get("VOICE_ID")
            audio = generate(text=response,model=model_id,voice=voice_id)
            
            # Convert the audio to bytes for Streamlit's audio widget
            audio_bytes = BytesIO(audio).read()

            # Play audio using Streamlit
            st.audio(audio_bytes, format='audio/ogg')

    except Exception as e:
        st.error(f"Error: {type(e).__name__}")
        st.error(str(e))
        st.text(traceback.format_exc())

