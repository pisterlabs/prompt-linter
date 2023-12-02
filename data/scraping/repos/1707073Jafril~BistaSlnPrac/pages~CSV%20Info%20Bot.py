#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 04:53:12 2023

@author: shbmsk
"""

import streamlit as st
import os
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from apiKey import openapi_key , serpapi_key

st.title("Bista Solution Inc")
st.subheader("Information Retrival From CSV")

os.environ['OPENAI_API_KEY'] = openapi_key
os.environ['SERPAPI_API_KEY'] = serpapi_key


llmBista = OpenAI(temperature= 0.8)





#st.write("Upload a CSV file and enter a query to get an answer.")
file =  st.file_uploader("Upload CSV file",type=["csv"])
if not file:
    st.stop()

#i = int(input())

#if(file(type=['excel'])):

    

data = pd.read_csv(file)
st.write("Data Preview:")
st.dataframe(data.head())
agent = create_pandas_dataframe_agent(
                    llmBista,data,verbose=True)
query = st.text_input("Enter a prompt : ") 

if st.button("Enter"):
    answer = agent.run(query)
    st.write("Result:")
    st.write(answer)





 



#if __name__ == "__main__":
    