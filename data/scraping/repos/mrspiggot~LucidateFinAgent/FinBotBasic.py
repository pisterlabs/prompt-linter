import streamlit as st
from PIL import Image
from langchain.chat_models import ChatOpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain import SerpAPIWrapper, WikipediaAPIWrapper
from langchain.agents.tools import Tool
from typing import Optional, Type
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logo = Image.open('static/ai_icon.png')
st.sidebar.title('Langchain Basic 🦜')
st.sidebar.image(logo, use_column_width=True)
temperature = st.sidebar.slider('Select a value', min_value=0.0, max_value=1.0, value=0.2, step=0.1)

models = ["gpt-4", "gpt-4-0613", "gpt-4-32k", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613"]
model = st.sidebar.selectbox("Select a GPT Model", models, index=0)

st.title('Basic FinBot Demo 🦜🔗')
user_input = st.text_input('Your financial question?')
enter_button = st.button('Enter')

if enter_button:
    if user_input:
        st.write(f'You entered: {user_input}, Model is: {model}, with temperature: {temperature}')
        model = ChatOpenAI(model=model, temperature=temperature)
        response = model.predict(user_input)
        st.write(response)

