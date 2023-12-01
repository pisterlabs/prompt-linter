import streamlit as st

import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from transformers import pipeline

from langchain.text_splitter import CharacterTextSplitter

import datetime

import os

import sys

from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import AutoTokenizer, AutoModelWithLMHead





def generate_response_t5_base(txt):

    tokenizer = AutoTokenizer.from_pretrained("t5-base") #english,french,romanian,german

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

    prompt = txt

    input = tokenizer(prompt, return_tensors="pt")

    output = model.generate(**input, max_length=200, num_beams=2, no_repeat_ngram_size=2, early_stopping=True)

    #print(tokenizer.decode(output[0]))

    return tokenizer.decode(output[0])





def generate_response_t5_large(txt):

    tokenizer = AutoTokenizer.from_pretrained("t5-large")

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")

    prompt = txt

    input = tokenizer(prompt, return_tensors="pt")

    output = model.generate(**input, max_length=200, num_beams=2, no_repeat_ngram_size=2, early_stopping=True)

    print(tokenizer.decode(output[0]))

    return tokenizer.decode(output[0])





def generate_response_t5_3b(txt):

    tokenizer = AutoTokenizer.from_pretrained("t5-3b")

    model = AutoModelWithLMHead.from_pretrained("t5-3b")

    prompt = txt

    input = tokenizer(prompt, return_tensors="pt")

    output = model.generate(**input, max_length=200, num_beams=2, no_repeat_ngram_size=2, early_stopping=True)

    print(tokenizer.decode(output[0]))

    return tokenizer.decode(output[0])





def generate_response_bloomz_1b7(txt):

    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-1b7")

    model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-1b7")

    prompt = txt

    input = tokenizer(prompt, return_tensors="pt")

    output = model.generate(**input, max_length=200, num_beams=2, no_repeat_ngram_size=2, early_stopping=True)

    print(tokenizer.decode(output[0]))

    return tokenizer.decode(output[0])





def generate_response_bloomz_3b(txt):

    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-3b")

    model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-3b")

    prompt = txt

    input = tokenizer(prompt, return_tensors="pt")

    output = model.generate(**input, max_length=200, num_beams=2, no_repeat_ngram_size=2, early_stopping=True)

    print(tokenizer.decode(output[0]))

    return tokenizer.decode(output[0])








############# Displaying images on the front end #################

st.set_page_config(page_title="Text Translation",

                   page_icon='./logo.png',

                   layout="wide",  #or wide

                   initial_sidebar_state="expanded",

                   menu_items={

                        'Get Help': 'https://docs.streamlit.io/library/api-reference',

                        'Report a bug': "https://www.extremelycoolapp.com/bug",

                        'About': "# This is a header. This is an *extremely* cool app!"

                                },

                   )





### HEADER section

image = "./logo.png"
st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
st.title("Text Translation")

#st.image('Headline.jpg', width=750)

from streamlit_extras.app_logo import add_logo
add_logo("logo.png", height=60)

option_input = st.selectbox("Input language", ['english'])


txt_input = st.text_area("Enter your text here", '', height=300,key = "original")


txt_input = """English: """ + txt_input

option1 = st.selectbox('Output language',

                       ('french','german'))

if option1 == 'german':
    txt_input = txt_input + """\nGerman:"""
else:
    txt_input = txt_input + """\nFrench:"""



result = []

with st.form('Translation_form', clear_on_submit=True):

    options = st.selectbox(

    'Choose a model for Translation:',

    options=["t5-base", "t5-large", "t5-3b", "bigscience/bloomz-1b7","bigscience/bloomz-3b"])

    st.write('You selected:', options)

    submitted = st.form_submit_button('Submit')

    response = ''

                     

    if submitted:

        with st.spinner('Initializing pipelines...'):

            st.success(' AI Translation started', icon="🆗")

           

        if options=="t5-base":

            start = datetime.datetime.now()

            response = response + generate_response_t5_base(txt_input)

            stop = datetime.datetime.now() #not used now but useful

            elapsed = stop - start

            st.success(f'Translation completed in {elapsed}', icon="🆗")

        elif options=="t5-large":

            start = datetime.datetime.now()

            response = response + generate_response_t5_large(txt_input)

            stop = datetime.datetime.now() #not used now but useful

            elapsed = stop - start

            st.success(f'Translation completed in {elapsed}', icon="🆗")

        elif options=="t5-3b":

            start = datetime.datetime.now()

            response = response + generate_response_t5_3b(txt_input)

            stop = datetime.datetime.now() #not used now but useful

            elapsed = stop - start

            st.success(f'Translation completed in {elapsed}', icon="🆗")

        elif options=="bigscience/bloomz-1b7":

            start = datetime.datetime.now()

            response = response + generate_response_bloomz_1b7(txt_input)

            stop = datetime.datetime.now() #not used now but useful

            elapsed = stop - start

            st.success(f'Translation completed in {elapsed}', icon="🆗")

        else:

            start = datetime.datetime.now()

            response = response + generate_response_bloomz_3b(txt_input)

            stop = datetime.datetime.now() #not used now but useful

            elapsed = stop - start

            st.success(f'Translation completed in {elapsed}', icon="🆗")

       

         

st.info(response)        