import streamlit as st
from streamlit_chat import message as st_message
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import os
import openai
import numpy as np
import time



pre_questions = [
    "What is the capital city of Australia?",
    "Who invented the first telescope?",
    "Which country is both the largest producer and consumer of chocolate?",
    "What is the longest river in South America?",
    "Who was the first president of the United States?",
    "What is the largest country in Africa by land area?"]

#Preparing the answer retrieval from HuggingFace

access_token = os.environ["HUGGINGFACE_API_TOKEN"]
#provide the prompt template
template = """
For every question asked, you should return the answer in this format

Examples:
Question - What is the captial city of Thailand?
Answer - The capital city of Thailand is Bangkok.

Question - What does NLP stand for?
Answer - NLP stands for 'Natural Language Processing'

Question - who is Emmanuel Macron?
Answer - He is the current President of France.


The answer should be a full sentence, not an incomplete noun phrase

Question: {question}

"""

prompt = PromptTemplate(template=template, input_variables=["question"])
flan_llm= HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0.5},
                    huggingfacehub_api_token = access_token )
flan_llm_chain = LLMChain(prompt=prompt, llm=flan_llm, verbose=True)


def generate_answer(question):
    answer = flan_llm_chain.run(question)
    return answer


st.write("## 💬 Conversation with prompt 2")
with st.expander("see the prompt"):
    st.write(template)
         



question = "How are you doing today?"


if "history_5" not in st.session_state:
    st.session_state.history_5 = []
    st.session_state.history_5.append({"message": "how can I help you today?", "is_user": False} )
    st.session_state.history_5.append({"message": question, "is_user": True} )


#  Define the callback functions to update the text input value
def update_input_value(button_text):
    # Update the input value to the text of the clicked button
    st.session_state.input_value_5 = button_text
    
    
def send_message():
    global button1_text
    st.session_state.history_5.append({"message": input_value  , "is_user": False})
    
    # to get new question
    time.sleep(2)
    question = np.random.choice(pre_questions)
    st.session_state.history_5.append({"message": question   , "is_user": True})
    # button1_text = 

# st.write(question)
# st.write(access_token)


if 'input_value_5' not in st.session_state:
    st.session_state.input_value_5 = ""

row1_col1, row1_col2 = st.columns([2,1])
row2_col1, row2_col2 = st.columns([2,1])

# button1_text, button2_text, button3_text = np.random.choice(a= pre_answers, size=3 ,replace=False)
button1_text = generate_answer(st.session_state.history_5[-1]['message'])

with row2_col2:
    st.markdown("---")
    st.write("#### Candidate answers:")
    button1_clicked = st.button(button1_text)
    
    
    # Call the callback function when a button is clicked
if button1_clicked:
    update_input_value(button1_text)
    
    
with row1_col1:
    st.write("#### Conversation Screen:")
    for chat in st.session_state.history_5:
        st_message(**chat)
        
with row2_col1:
    st.markdown("---")
    st.write("#### Agent Placeholder:")
    # Update the default value of the text input widget
    input_value = st.text_input('Enter a value',
                                st.session_state.input_value_5)
    send_message_button = st.button("Send the message",
                                    on_click=send_message)