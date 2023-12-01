# local run
# from config import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, OPENAI_API_KEY
import os
import openai
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from streamlit_chat import message
from utils import set_background, generate_response

import streamlit as st
st.set_page_config(page_title="PhilosopherGPT", 
                   page_icon=":robot_face:")

set_background('assets/background.png')
st.markdown("<h1 font-family:trajan; style='text-align: center;'>PhilosopherGPT</h1>", unsafe_allow_html=True)

# Set org ID and API key
openai.api_key = st.secrets['OPENAI']['openai_api_key']

qdrant_client = QdrantClient(
    url=st.secrets['QDRANT']['host'],
    port=st.secrets['QDRANT']['port'],
    api_key=st.secrets['QDRANT']['qdrant_api_key'],
)

@st.cache_resource(ttl=24*3600, hash_funcs={"MyUnhashableClass": lambda _: None})
def load_model():
	  return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

retrieval_model = load_model()

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'references' not in st.session_state:
    st.session_state['references'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

model_name = "GPT-3.5"

# Map model names to OpenAI model IDs
if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    print('Enter a model')

# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output, references, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input, model)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state['references'].append(references)
        st.session_state['model_name'].append(model_name)
        st.session_state['total_tokens'].append(total_tokens)

        # from https://openai.com/pricing#language-models
        if model_name == "GPT-3.5":
            cost = total_tokens * 0.002 / 1000
        else:
            cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

        st.session_state['cost'].append(cost)
        st.session_state['total_cost'] += cost

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i]+'\n\nRéférences:\n'+st.session_state["references"][i], key=str(i)) #
            st.write(
                f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            #counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")

clear_button = st.button("Clear Conversation", key="clear")

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    #counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
