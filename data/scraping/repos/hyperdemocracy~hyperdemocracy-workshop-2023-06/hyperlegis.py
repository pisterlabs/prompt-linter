import streamlit as st
ss = st.session_state
import hyperdemocracy as hd
from langchain.callbacks.streamlit import StreamlitCallbackHandler

import os
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import ChatOpenAI


st.title('📜✨ Hyperlegis - Ask Questions About Legislation')

user_openai_api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password", help="Set this to run your own custom questions.", key="openai_api_key"
)
user_serpapi_api_key = st.sidebar.text_input(
    "SerpAPI API Key",
    type="password",
    help="Set this to run your own custom questions. Get yours at https://serpapi.com/manage-api-key.",
)

if user_openai_api_key or user_serpapi_api_key:
    openai_api_key = user_openai_api_key
    serpapi_api_key = user_serpapi_api_key
    enable_custom = True
# else:
#     openai_api_key = st.secrets["openai_api_key"]
#     serpapi_api_key = st.secrets["serpapi_api_key"]
#     enable_custom = False

llm = ChatOpenAI(model_name="gpt-4", request_timeout=120, temperature = 0, openai_api_key=ss.get('openai_api_key'))

tab1, tab2 = st.tabs(["LegisQA", "HyperdemocracyAgent"])


with tab1:
    qaws = hd.get_qa_with_sources_chain(llm)
    query = st.text_input("Enter your question here","", key="qaws")
    kdocs = st.slider("Number of source documents to return", 1, 10, 2)
    if query != "":

        out = qaws(query)

        st.subheader('Answer')
        st.write(out['answer'])
        st.subheader(out['sources'])

with tab2:
    st.write('implement hyperdemocracy agent here')
    # agent = hd.get_agent(llm)

    # key = "input"
    # shadow_key = "_input"

    # if key in st.session_state and shadow_key not in st.session_state:
    #     st.session_state[shadow_key] = st.session_state[key]

    # with st.form(key="form"):
    #     agent_input = st.text_input("Or, ask your own question", key=shadow_key)
    #     st.session_state[key] = agent_input
    #     submit_clicked = st.form_submit_button("Submit Question")

    # question_container = st.empty()
    # results_container = st.empty()  

    # from clear_results import with_clear_container

    # if with_clear_container(submit_clicked):
    #     res = results_container.container()
    #     streamlit_handler = StreamlitCallbackHandler(
    #         parent_container=res,
    #         max_thought_containers=4,
    #         expand_new_thoughts=True,
    #         collapse_completed_thoughts=True,
    #     )

    #     answer = agent.run(agent_input, callbacks=[streamlit_handler])
    #     res.write(f"**Answer:** {answer}")