# streamlit run file.py

import streamlit as st
from streamlit_chat import message
from bardapi import Bard
import os
from dotenv import load_dotenv
load_dotenv() 
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


from langchain.llms import OpenAI
llm = OpenAI(temperature=0.9)

def generate_response(prompt):
    # token = 'Xwj6qHKoY50eHPoKX_v1aI1Q1ikX4bNQM9TsIxP4AjSQ4eMJShBBOECVD0iB5Fl9dTxk0A.'
    # bard = Bard(token=token)
    # response = bard.get_answer(prompt)['content']
    # return response
    prompt = "You are a AI bot that Replicated MS Dhoni . Give advises in his form , with his leadership experise. You help people be their best and guide them like him.Keep it consise and not long.Add humour of dhoni also" + prompt + "\nAI Bot:"
    return llm(prompt)

#st.title('Demo')
def get_text():
    input_text = st.text_input("Dhoni Bot:","Hey wassup",key = 'input')
    return input_text

st.title('Dhoni GPT')

changes = '''
<style>
[data-testid="stAppViewContainer"] {

    background-image: url("https://images.unsplash.com/photo-1625401586082-9a9b17bc4ce5?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTB8fGNyaWNrZXQlMjBpbmRpYXxlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&w=500&q=60");
    background-size: cover;
    }
    
    div.esravye2 > iframe{
    background-color: transparent;
    }

</style>
'''

st.markdown(changes, unsafe_allow_html=True)
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

user_input = get_text()
if user_input:
    print(user_input)
    output = generate_response(user_input)
    print(output)
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1,-1,-1):
        message(st.session_state['generated'][i],key= str(i))
        message(st.session_state['past'][i],key= "user_" + str(i),is_user=True)