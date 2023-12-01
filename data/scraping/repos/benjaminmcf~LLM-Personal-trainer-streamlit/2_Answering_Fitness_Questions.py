import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

import json
import openai

def generate_answer():
    generate_button.empty()
    st.session_state.generate = True

# Load fitness questions
with open('questions.json', 'r') as json_file:
    json_data = json_file.read()
    data = json.loads(json_data)


#  Set page title
st.set_page_config(
    page_title="Answering fitness questions",
)

# title of content
st.title("LLM Personal Trainer")
st.header("Answering Fitness Questions")


# Initialise the session states if they dont exist
if "generate" not in st.session_state:
    st.session_state.generate = False

if 'currentkey' not in st.session_state:
     st.session_state.currentkey = ''


try:
    st.session_state.currentkey = st.secrets["open_ai_key"]
except:
    pass

openai.api_key = st.session_state.currentkey
 
questions = data['questions']
question_list = []


def validate():
    try:
        text_input = st.session_state.input
        openai.api_key = text_input
        response = openai.Completion.create(
            engine="davinci",
            prompt="validating openaikey",
            max_tokens=5
        )
        st.session_state.currentkey = text_input
    except:
        side_validation = st.sidebar.text('OPEN AI API key not valid')



with st.sidebar.form('Enter OPEN API key'):
    st.text_input("Enter open api key",key='input')
    st.form_submit_button('Validate key', on_click=validate)


if st.session_state.currentkey:
    side_text = st.sidebar.text(
    f'Current OPEN AI API Key is valid'
    )




if st.session_state.currentkey:
    question_list.append('Select a question')
    for question in questions:
        question_list.append(question['question'])

    option = st.selectbox('Select a question',question_list,placeholder='Select a question')

    if option != 'Select a question':

        
        generate_button = st.empty()
        generate_button.button("generate answer",type='primary',on_click=generate_answer)
        if st.session_state.generate:
            with st.spinner("Answering your question..."):
                llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0.5,openai_api_key=st.session_state.currentkey)
                template = """
                {option}
                """
                promp = PromptTemplate(
                    input_variables=['option'],
                    template=template
                )
                chain = LLMChain(llm=llm, prompt=promp)
                output = chain.run({'option':option})
            st.write(output)
            st.session_state.generate = False

else:
    st.header('Enter your Open AI API key to use functionality')
