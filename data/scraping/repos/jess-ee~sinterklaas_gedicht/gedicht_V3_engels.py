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


# Set the Eleven Labs and OpenAI API Key 
set_api_key(os.environ.get("ELEVEN_LABS_API_KEY"))

apikey = os.getenv('OPENAI_API_KEY')

with open('Hobbies_English.csv', 'r') as f:
    hobbies_options = f.read().splitlines()
with open('Badtraits_English.csv', 'r') as f:
    traits_options = f.read().splitlines()

#App framework
st.title('Coolblue Saint Nicholas poems ✍️')

st.markdown("""
    Welcome to the Cooblue poem generator!

    """)

name = st.text_input('For who is this gift?')
hobby = st.multiselect('What are his or here hobbies? (select at least one option)', hobbies_options, max_selections=2)
traits = st.multiselect('What are his or her bad habits? (slect at leat one option)',traits_options,max_selections=2)
product_type_name = st.text_input('Which gift have you bought for him or her?')
product = st.text_area('Fill in some of the product information')

#Chatmodel 

chat_model= ChatOpenAI(temperature=0.6, model="gpt-4")

#Prompt template

system_message_prompt = SystemMessagePromptTemplate.from_template("""You are writing Saint Nicholas poems for the customers of Coolblue.

Write the poems based on information about the customer and the product they have purchased.

The poem should be funny, positive, and cheerful. Don't reveal the product, but dance around it.

Use a maximum of 8 lines.

Respond with "You're going back to spain with Saint Nicholas" when someone puts in an offensive name.

""")
human_message_prompt = HumanMessagePromptTemplate.from_template("""Informatie about the customer:
- Name: {name}
- Hobbies: {hobby}
- Bad habits {traits}

Information about the product:
- {product_type_name}
- {product}
""")
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

#LLM Chain

gedicht_chain = LLMChain(llm=chat_model, prompt=chat_prompt, verbose = True)

# show stuff
if st.button('Ask G-Piet-R for a poemSt!'):
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


    




