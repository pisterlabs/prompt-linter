import streamlit as st
from streamlit_extras.stateful_button import button
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
import sys
import time as time
from top_n_tool import run_tool

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

if prompt := st.chat_input("Send a message"):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        response = llm.call_as_llm(prompt)
        st.write(f"**Question:** {prompt}")
        st.write(f"**Answer:** {response}")
        
if "feedback" not in st.session_state:
    st.session_state.feedback = []


# if button('Feedback', key="open_feedback"):
#     feedback_text = st.text_input("Please provide your feedback")
#     feedback_score = st.number_input("Rate your experience (0-10)", min_value=0, max_value=10)
#     user_feedback = pd.DataFrame({"Feedback_Text": [feedback_text], "Feedback_Score": [feedback_score]})
#     if button('Send', key="send_feedback"):
#         if os.path.exists("user_feedback.csv"):
#             user_feedback.to_csv("user_feedback.csv", mode='a', header=False, index=False)
#         else:
#             user_feedback.to_csv("user_feedback.csv", index=False)


def b_get_feedback():
    if button('Feedback', key="open_feedback"):
        feedback_text = st.text_input("Please provide your feedback")
        feedback_score = st.number_input("Rate your experience (0-10)", min_value=0, max_value=10)
        user_feedback = pd.DataFrame({"Feedback_Text": [feedback_text], "Feedback_Score": [feedback_score]})
        if button('Send', key="send_feedback"):
            if os.path.exists("user_feedback.csv"):
                user_feedback.to_csv("user_feedback.csv", mode='a', header=False, index=False)
            else:
                user_feedback.to_csv("user_feedback.csv", index=False)
            time.sleep(1)
            st.toast("✔️ Feedback received! Thanks for being in the loop 👍\nClick the `Feedback` button to open or close this anytime.")


feedback_button = b_get_feedback()

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
