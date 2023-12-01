# import openai
# import streamlit as st


# st.title("Isolated Falcons")


# openai.api_key =  st.secrets["OPENAI_API_KEY"]

# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"


# # with st.chat_message(name="user",avatar="👦"):
# #     st.write("Hello :wave:")   

# #history starts
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# #history session
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])


# #chat input
# if prompt := st.chat_input("What's Up ?"):
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     # with st.spinner("Thinking..."):    
#     st.session_state.messages.append({"role": "user", "content": prompt})


#     # response = f"Echo: {prompt}"
        
#     # with st.chat_message("assistant",avatar="🤖"):
#     #      st.markdown(response)

#     # st.session_state.messages.append({"role": "assistant", "content": response})

#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = ""
#         for response in openai.ChatCompletion.create(
#             model=st.session_state["openai_model"],
#             messages=[
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages
#                 ],
#                 stream = True,
#         ):
#             full_response += response.choices[0].delta.get("content","")
#             message_placeholder.markdown(full_response + " ")
#         message_placeholder.markdown(full_response)
#         st.session_state.messages.append({"role": "assistant", "content": full_response})


#New UI

# import openai
# import streamlit as st
# from streamlit_chat import message

# openai.api_key =  st.secrets["OPENAI_API_KEY"]

# def generate_response(prompt):
#     completions = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         temperature=0.5,
#         max_tokens=1024,
#         n = 1,
#         stop  = None,
#     )

#     message = completions.choices[0].text
#     return message

# st.title("Isolated Falcons")

# st.write("""
#     ## Chatbot
# """
# )

# if 'generated' not in st.session_state:
#     st.session_state['generated'] = []

# if 'past' not in st.session_state:
#     st.session_state['past'] = []

# def get_text():
#     input_text= st.text_input("You :","What's Up ?",key="input")
#     return input_text

# user_input = get_text()

# if user_input:
#     output = generate_response(user_input)
#     st.session_state.past.append(user_input)
#     st.session_state.generated.append(output)

# if st.session_state.generated:

#     for i in range(len(st.session_state.generated)-1,-1,-1):
#         message(st.session_state.generated[i],key=str(i))
#         message(st.session_state.past[i],is_user=True,key=str(i) + '_user')

import streamlit as st
from streamlit_chat import message
########################################________LANGCHAIN________####################
import os

###############################################____LLM____################
# Using OPENAI LLM's
from langchain.llms import OpenAI
# Creating Prompt Templates
from langchain.prompts import PromptTemplate
# Creating Chains
from langchain.chains import LLMChain
def query(payload):
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(input_variables=["Product"],
                            template="{Product}")
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(payload["inputs"]["text"])
    # respone is a string
    return response 
###############################################____CHAT MODEL____################
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
def chatquery(payload):
    chat = ChatOpenAI(temperature=0, streaming=True)

    template="You are a helpful assistant"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(text=payload["inputs"]["text"])
    # Result is a string
    return result

st.set_page_config(
    page_title="Assistant",
    page_icon=":robot:"
)

st.header("Welcome to Assistant")

# state to hold generated output of llm
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

# state to hold past user messages
if 'past' not in st.session_state:
    st.session_state['past'] = []

# streamlit text input
def get_text():
    input_text = st.text_input("Input Message: ","", key="input")
    return input_text 

user_input = get_text()

# check if text input has been filled in
if user_input:
    # run langchain llm function returns a string as output
    output = chatquery({
        "inputs": {
            "past_user_inputs": st.session_state.past,
            "generated_responses": st.session_state.generated,
            "text": user_input,
        },"parameters": {"repetition_penalty": 1.33},
    })

    # append user_input and output to state
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# If responses have been generated by the model
if st.session_state['generated']:
    # Reverse iteration through the list
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        # message from streamlit_chat
        message(st.session_state['past'][::-1][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][::-1][i], key=str(i))

# I would expect get_text() needs to be called here as a callback
# But i have issues with retreving user_input