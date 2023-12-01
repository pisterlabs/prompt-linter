import openai

# from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
# from langchain.memory import ConversationBufferWindowMemory

import streamlit as st
from streamlit_chat import message

import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

K = 10 # Number of previous convo

template = """Hope is an expert in performing Cognitive Behavioural Therapy. Hope will be the Users Therapist.
Hope will converse with the user and help the user to overcome their mental health problems. Hope is very experienced and keeps in mind previous conversations made with the user.
User will share their thoughts and problems with Hope and Hope will try and solve them by Cognitive Behavioural Therapy.
Hope can help users who struggle with anxiety, depression, trauma, sleep disorder, relationships, work-stress, exam-stress and help them.
Hope may also suggest breathing exercises or simple tasks or any other conventional methods that may help the User.

{chat_history}
User: {human_input}
Hope:"""


def generate_response(inp,temp):
    # prompt = PromptTemplate(
    # input_variables=["history", "human_input"], 
    # template=template
    # )
    history = ""
    for i in range(len(st.session_state.generated)):
        history = history+"User: "+st.session_state.past[i]+"\n"
        history = history+"Hope: "+st.session_state.generated[i]+"\n"
    prompt = temp.format(chat_history = history,human_input = inp)

    print(prompt)
    response = openai.Completion.predict(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0,
    )
    
    message = response.choices[0].text
    return message

#Creating the chatbot interface
st.title("H O P E")

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []
    
# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("You: ","", key="input")
    return input_text

user_input = get_text()

if user_input:
    output = generate_response(user_input,template)
    # store the output 
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
    print(st.session_state.generated)
    print(st.session_state.past)
    

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
