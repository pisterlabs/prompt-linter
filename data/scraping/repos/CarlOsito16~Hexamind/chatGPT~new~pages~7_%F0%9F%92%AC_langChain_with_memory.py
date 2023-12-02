import streamlit as st
from streamlit_chat import message as st_message
from langchain import PromptTemplate
from langchain import OpenAI, ConversationChain, LLMChain


from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub

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
# template = """
# The following is a friendly conversation between a human and an AI.
# The AI is talkative and provides lots of specific details from its context. 
# If the AI does not know the answer to a question, it truthfully says it does not know.

# {chat_history}
# Human: {question}
# Bot:
# """


template = """
The following is a friendly conversation between a human and an AI.
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.
AI does not know anything personal about the human until given the informattion about it.
AI must remember and learn from {chat_history}

Human: {human_input}
AI:"""



PROMPT = PromptTemplate(
    input_variables=["chat_history", "human_input"], 
    template=template
)
MEMORY = ConversationBufferMemory(memory_key="chat_history")



flan_llm= HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0.6},
                    huggingfacehub_api_token = access_token )
# flan_llm_chain = LLMChain(prompt=prompt, llm=flan_llm, verbose=True)
# bigscience/bloom

conversation = LLMChain(
    llm=flan_llm, 
    prompt = PROMPT,
    verbose=True, 
    memory=MEMORY
)





#Change the method to generate the answer
def generate_answer(question):
    answer = conversation.predict(human_input = question)
    return answer


st.write("## 💬 Conversation with prompt with memory")
with st.expander("see the prompt"):
    st.write(template)
         



question = "How are you doing today?"


if "history_7" not in st.session_state:
    st.session_state.history_7 = []
    st.session_state.history_7.append({"message": "how can I help you today?", "is_user": False} )
    st.session_state.history_7.append({"message": question, "is_user": True} )
    # memory.save_context({"input" : st.session_state.history_7[-2]['message']}, 
    #                     {"output" : st.session_state.history_7[-1]['message']})

# st.write(st.session_state.history_7[-2]['message'])

#  Define the callback functions to update the text input value
def update_input_value(button_text):
    # Update the input value to the text of the clicked button
    st.session_state.input_value_7 = button_text


    
def send_message():
    global button1_text, question_index
    st.session_state.history_7.append({"message": input_value  , "is_user": False})
    
    # to get new question
    time.sleep(2)
    question = pre_questions[st.session_state.memory]
    st.session_state.history_7.append({"message": question   , "is_user": True})
    st.session_state.memory += 1
    # memory.save_context({"input" : st.session_state.history_7[-4, -2]['message']}, 
    #                     {"output" : st.session_state.history_7[-3, -1]['message']})
    # button1_text = 

# st.write(question)
# st.write(access_token)


if 'input_value_7' not in st.session_state:
    st.session_state.input_value_7 = ""
    
if 'memory' not in st.session_state:
    st.session_state.memory = 0

row1_col1, row1_col2 = st.columns([2,1])
row2_col1, row2_col2 = st.columns([2,1])

# button1_text, button2_text, button3_text = np.random.choice(a= pre_answers, size=3 ,replace=False)
button1_text = generate_answer(st.session_state.history_7[-1]['message'])

with row2_col2:
    st.markdown("---")
    st.write("#### Candidate answers:")
    button1_clicked = st.button(button1_text)
    
    
    # Call the callback function when a button is clicked
if button1_clicked:
    update_input_value(button1_text)
    
    
with row1_col1:
    st.write("#### Conversation Screen:")
    for chat in st.session_state.history_7:
        st_message(**chat)
        
with row2_col1:
    st.markdown("---")
    st.write("#### Agent Placeholder:")
    # Update the default value of the text input widget
    input_value = st.text_input('Enter a value',
                                st.session_state.input_value_7)
    send_message_button = st.button("Send the message",
                                    on_click=send_message)
    
    


st.write(conversation.predict(human_input = 'what is my name?'))

st.write(MEMORY.load_memory_variables({}))


st.write(conversation.predict(human_input = 'My name is Poon. Nice to meet you'))

st.write(MEMORY.load_memory_variables({}))

st.write(conversation.predict(human_input = 'What is my name?'))

st.write(MEMORY.load_memory_variables({}))