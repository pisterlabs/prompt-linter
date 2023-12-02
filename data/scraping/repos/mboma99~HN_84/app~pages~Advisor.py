import pandas as pd
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
import os
from apikey import apikey
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from FinBot import FinBot

st.title('Stock Advisor')

stocks = pd.read_csv("nasdaq.csv")
stocks_symbols = stocks["Symbol"].tolist()

stocks_input = st.text_input(
    'Enter stock symbol:'
)

selected_stocks = [
    symbol.strip().upper()
    for symbol in stocks_input.split(',')
    if symbol.strip().upper() in stocks_symbols
]

if not selected_stocks:
    st.warning('Please enter a valid stock symbol.')

else:

    model = FinBot()

    links = []
    for symbols in selected_stocks:
        links = model.getSearchResults(f"Latest {symbols} news")
        st.header(f"Who are  {symbols}?")
        st.write(f"{model.getInsights(symbols)}")
        st.subheader(f"learn more about {symbols} by clicking the links below:")
        for link in links:
            st.write(link)


st.header("FinBot: The Conversational Finance Chat Bot")
os.environ['OPENAI_API_KEY'] = apikey
chat = ChatOpenAI(temperature=0)
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="act as a stock advisor. keep your answers short and concise")
    ]

with st.sidebar:
    user_input = st.text_input("your message: ", key="user_input")

if user_input:
    #message(user_input, is_user=True)
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.spinner("Thinking..."):
        response = chat(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))
    #message(response.content, is_user=False)

messages = st.session_state.get('messages',[])
for i, msg in enumerate (messages[1:]):
    if i % 2 == 0:
        message(msg.content, is_user=True, key=str(i)+'_user')
    else:
        message(msg.content, is_user=False, key=str(i)+'_ai')



    #prompt = st.text_area('Chat with the Chat Bot', height=75)
    #if prompt:
    #    response = llm(prompt)
    #    st.write(response)
