import streamlit as st
from streamlit_chat import message as st_message
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
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
openai.api_key = os.environ["OPENAI_API_KEY"]
#provide the prompt template
template = """
The following is a friendly conversation between a human and an AI.
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.
AI knows NOTHING anything personal about the human until given the informattion about it.
AI must remember and learn from {chat_history}

Human: {human_input}
Assistant:
"""


PROMPT = PromptTemplate(
    input_variables=["chat_history", "human_input"], 
    template=template
)


MEMORY = ConversationBufferMemory(memory_key="chat_history")

chatgpt_chain = LLMChain(
    llm=OpenAI(temperature=0.5), 
    prompt=PROMPT, 
    verbose=True, 
    memory=MEMORY,
)



st.write(chatgpt_chain.predict(human_input = 'who are you?'))

st.write(MEMORY.load_memory_variables({}))



st.write(chatgpt_chain.predict(human_input = 'what is my name?'))

st.write(MEMORY.load_memory_variables({}))


st.write(chatgpt_chain.predict(human_input = 'My name is Poon. Nice to meet you'))

st.write(MEMORY.load_memory_variables({}))

st.write(chatgpt_chain.predict(human_input = 'What is my name?'))

st.write(MEMORY.load_memory_variables({}))



#Change the method to generate the answer
def generate_answer(question):
    answer = chatgpt_chain.predict(human_input = question)
    return answer


st.write("## 💬 Conversation with prompt with memory")
with st.expander("see the prompt"):
    st.write(template)
         


question = "How are you doing today?"


if "history_8" not in st.session_state:
    st.session_state.history_8 = []
    st.session_state.history_8.append({"message": "how can I help you today?", "is_user": False} )
    st.session_state.history_8.append({"message": question, "is_user": True} )


#  Define the callback functions to update the text input value
def update_input_value(button_text):
    # Update the input value to the text of the clicked button
    st.session_state.input_value_8 = button_text


    
def send_message():
    global button1_text, question_index
    st.session_state.history_8.append({"message": input_value  , "is_user": False})
    
    # to get new question
    time.sleep(2)
    question = pre_questions[st.session_state.memory_8]
    st.session_state.history_8.append({"message": question   , "is_user": True})
    st.session_state.memory_8 += 1
    # button1_text = 

# st.write(question)
# st.write(access_token)



if 'input_value_8' not in st.session_state:
    st.session_state.input_value_8 = ""
    
    
if 'memory_8' not in st.session_state:
    st.session_state.memory_8 = 0
    

row1_col1, row1_col2 = st.columns([2,1])
row2_col1, row2_col2 = st.columns([2,1])

# button1_text, button2_text, button3_text = np.random.choice(a= pre_answers, size=3 ,replace=False)
st.write(st.session_state.history_8[-1]['message'])
button1_text = generate_answer(st.session_state.history_8[-1]['message'])

# st.write(memory.load_memory_variables({})['history'][-1].content)


with row2_col2:
    st.markdown("---")
    st.write("#### Candidate answers:")
    button1_clicked = st.button(button1_text)
    

    
    
    # Call the callback function when a button is clicked
if button1_clicked:
    st.write('click click')
    # update_input_value(button1_text)
    st.write('nothig happened')
    
with row1_col1:
    st.write("#### Conversation Screen:")
    for chat in st.session_state.history_8:
        st_message(**chat)
        
with row2_col1:
    st.markdown("---")
    st.write("#### Agent Placeholder:")

    # Update the default value of the text input widget
    input_value = st.text_input('Enter a value',
                                st.session_state.input_value_8)
    send_message_button = st.button("Send the message",
                                    on_click=send_message)
    
    
    
