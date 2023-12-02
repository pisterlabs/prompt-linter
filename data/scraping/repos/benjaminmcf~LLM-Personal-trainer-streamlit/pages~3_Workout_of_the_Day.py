import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

import openai

def generate_answer():
    generate_button.empty()
    st.session_state.generate = True

# Set page title
st.set_page_config(
    page_title="Workout of the day",
)

preference_list = [
    'Select a preference',
    'no preference',
    'endurance focus',
    'strength focus',

]

# Initialise the session state variables if they dont exist
if "generate" not in st.session_state:
    st.session_state.generate = False

if 'currentkey' not in st.session_state:
     st.session_state.currentkey = ''

if 'validate' not in st.session_state:
    st.session_state.validate = False

if 'validate_count' not in st.session_state:
    st.session_state.validate_count = 0


try:
    st.session_state.currentkey = st.secrets["open_ai_key"]
except:
    pass

openai.api_key = st.session_state.currentkey


st.title("LLM Personal Trainer")
st.header("Generate a Workout of the Day")

def validate():
    try:
        text_input = st.session_state.input
        openai.api_key = text_input
        st.session_state.validate_count = st.session_state.validate_count + 1
        response = openai.Completion.create(
            engine="davinci",
            prompt="validating openaikey",
            max_tokens=5
        )
        st.session_state.currentkey = text_input
        st.session_state.validate = False
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
    option = st.selectbox('Select a preference',preference_list,placeholder='Select a preference')

    if option != 'Select a preference':

        generate_button = st.empty()
        generate_button.button("generate workout of the day",type='primary',on_click=generate_answer)
        if st.session_state.generate:
            with st.spinner("Generating a random workout for you..."):
                llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0.7,openai_api_key=st.session_state.currentkey)
                template = """
                Create a random workout of the day, consider the following preference that the athlete has:
                {preference}
                """
                promp = PromptTemplate(
                    input_variables=['preference'],
                    template=template
                )
                chain = LLMChain(llm=llm, prompt=promp)
                output = chain.run({'preference':option})
            st.write(output)
            st.session_state.generate = False

else:
    st.header('Enter your Open AI API key to use functionality')
