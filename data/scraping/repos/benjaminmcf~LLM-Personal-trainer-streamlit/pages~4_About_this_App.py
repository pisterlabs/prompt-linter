import streamlit as st
import openai

st.set_page_config(
    page_title="About this app",
)


if "generate" not in st.session_state:
    st.session_state.generate = False


if 'openaikey' not in st.session_state:
    st.session_state.openaikey = '' 

if 'currentkey' not in st.session_state:
     st.session_state.currentkey = ''


try:
    st.session_state.currentkey = st.secrets["open_ai_key"]
except:
    pass

openai.api_key = st.session_state.currentkey

def validate():
    try:
        text_input = st.session_state.input
        openai.api_key = text_input
        response = openai.Completion.create(
            engine="davinci",
            prompt="validating openaikey",
            max_tokens=5
        )
        st.session_state.openaikey = text_input
        st.session_state.currentkey = text_input
    except:
        side_validation = st.sidebar.text('OPEN AI API key not valid')



with st.sidebar.form('Enter OPEN API key'):
    st.text_input("Enter open api key",key='input')
    st.form_submit_button('Validate key', on_click=validate)

if st.session_state.currentkey:
    side_text = st.sidebar.text(
    f'Current OPEN AI Key is valid'
    )

st.title('About this app')
st.write('This is a LLM based personal training fitness app. Developed as part of streamlit hackathon, September 2023')
st.write('Github repository for this application can be found here -> https://github.com/benjaminmcf/LLM-Personal-trainer-streamlit')
st.write('You can view the related medium blog post here -> https://medium.com/@mcfaddenrbenjamin/building-a-llm-personal-trainer-with-streamlit-langchain-337a8efac832')