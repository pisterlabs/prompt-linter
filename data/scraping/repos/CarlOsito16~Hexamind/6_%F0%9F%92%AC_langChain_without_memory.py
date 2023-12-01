import streamlit as st
from streamlit_chat import message as st_message
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import os
import openai
import numpy as np
import time



pre_questions = [
    "Where am I interning at?",
    "I am interning at Hexamind in Paris since the beginning of previous February",
    "So, where am I interning at",
    "That's all for today"]

#Preparing the answer retrieval from HuggingFace

access_token = os.environ["HUGGINGFACE_API_TOKEN"]
#provide the prompt template
template = """
The following is a friendly conversation between a human and an AI.
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.

Any personal question from the user, AI would not be able to answer until given a more detail by human.


{question}
"""

prompt = PromptTemplate(template=template, input_variables=["question"])
flan_llm= HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0.5},
                    huggingfacehub_api_token = access_token )
flan_llm_chain = LLMChain(prompt=prompt, llm=flan_llm, verbose=True)


def generate_answer(question):
    answer = flan_llm_chain.run(question)
    return answer


st.write("## 💬 Conversation with prompt with memory")
with st.expander("see the prompt"):
    st.write(template)
         



question = "How are you doing today?"


if "history_6" not in st.session_state:
    st.session_state.history_6 = []
    st.session_state.history_6.append({"message": "how can I help you today?", "is_user": False} )
    st.session_state.history_6.append({"message": question, "is_user": True} )


#  Define the callback functions to update the text input value
def update_input_value(button_text):
    # Update the input value to the text of the clicked button
    st.session_state.input_value_6 = button_text


    
def send_message():
    global button1_text, question_index
    st.session_state.history_6.append({"message": input_value  , "is_user": False})
    
    # to get new question
    time.sleep(2)
    question = pre_questions[st.session_state.memory]
    st.session_state.history_6.append({"message": question   , "is_user": True})
    st.session_state.memory += 1
    # button1_text = 

# st.write(question)
# st.write(access_token)


if 'input_value_6' not in st.session_state:
    st.session_state.input_value_6 = ""
    
if 'memory' not in st.session_state:
    st.session_state.memory = 0

row1_col1, row1_col2 = st.columns([2,1])
row2_col1, row2_col2 = st.columns([2,1])

# button1_text, button2_text, button3_text = np.random.choice(a= pre_answers, size=3 ,replace=False)
button1_text = generate_answer(st.session_state.history_6[-1]['message'])

with row2_col2:
    st.markdown("---")
    st.write("#### Candidate answers:")
    button1_clicked = st.button(button1_text)
    
    
    # Call the callback function when a button is clicked
if button1_clicked:
    update_input_value(button1_text)
    
    
with row1_col1:
    st.write("#### Conversation Screen:")
    for chat in st.session_state.history_6:
        st_message(**chat)
        
with row2_col1:
    st.markdown("---")
    st.write("#### Agent Placeholder:")
    # Update the default value of the text input widget
    input_value = st.text_input('Enter a value',
                                st.session_state.input_value_6)
    send_message_button = st.button("Send the message",
                                    on_click=send_message)