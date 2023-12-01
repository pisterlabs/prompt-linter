import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler


import llmate_config
llmate_config.general_config()

if ('openai_api_key' not in st.session_state) or (st.session_state['openai_api_key'] == ''):
    st.error('Please load OpenAI API KEY and connect to a database', icon='🚨')
else:
    st.subheader("Test your Agent")
    st.markdown(
    """
    **Why test the Agent with random questions? 🎠**

    Think of it like a pop quiz for a student. By tossing random questions, you gauge the Agent's adaptability and breadth of knowledge.

    - Uncover the Agent's strengths and quirks.
    - Familiarize yourself with its capabilities.
    - Learn how to phrase queries for optimal answers.

    
    Dive in and get playful! The best way to understand is to explore 🕵️‍♂️.

    """
)


    user_query = st.text_input("Question for the agent:")
    st_callback = StreamlitCallbackHandler(st.container())
    if user_query:
        response = st.session_state['sql_agent'].run(user_query, callbacks=[st_callback])
        st.write(response)