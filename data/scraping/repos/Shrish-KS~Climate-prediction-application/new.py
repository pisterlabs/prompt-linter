"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import altair as alt
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import create_csv_agent
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openai

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(openai_api_key=st.secrets["api_key"],temperature=0)
    chain = ConversationChain(llm=llm)
    return chain

chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Chat Model", page_icon=":robot:")
st.subheader("_LANGCHAIN CHAT MODEL_")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

st.subheader(":blue[_WELCOME BACK WHAT WOULD YOU LIKE TO DO:_]")

with st.expander("1.CHAT BOT"):
        input_text = st.text_input("You: ", "Hello, how are you?", key="input")
        user_input = input_text
        if user_input:
            output = chain.run(input=user_input)

            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

        if st.session_state["generated"]:

            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

    

with st.expander("_2.ANALYSIS_"):
    genre = st.radio(
    "How do You Want to Analyse",
    ('1.From a Dataset Which you Want', '2.From a Dataset which is Available'))

    if genre == "1.From a Dataset Which you Want":
        # data = pd.read_excel("environment.xlsx")
        # chart = alt.Chart(data).mark_line().encode(x='Element', y='Area')
        # st.altair_chart(chart, use_container_width=True)
        openai.api_key = st.secrets["api_key"]
        model_engine = "text-davinci-002"

        # Define the initial prompt for the OpenAI model
        model_prompt = "The answer to your question is:"

        # Create a file uploader widget for Excel files
        excel_file = st.file_uploader("Upload Excel file", type=["xlsx"])

        # If a file is uploaded, load it into a pandas dataframe
        if excel_file is not None:
            df = pd.read_excel(excel_file)

            # Extract the column names from the dataframe
            # Extract the data you need from the dataframe
            input_data = df

            # Define a function to generate OpenAI responses
            def generate_answer(input_str):
                prompt_str = model_prompt + " " + input_str
                response = openai.Completion.create(
                    engine=model_engine,
                    prompt=prompt_str,
                    temperature=0.5,
                    max_tokens=1024,
                    n=1,
                    stop=None,
                    timeout=60,
                )
                return response.choices[0].text.strip()

            # Allow the user to ask questions about the data
            question = st.text_input("Ask a question")

            # If a question is asked, generate an OpenAI response
            if question:
                # Construct the prompt for the OpenAI model
                prompt = model_prompt + " "
                for input_str in input_data:
                    prompt += f"Q: What is the {question} for {input_str}? A: "
                
                # Generate the OpenAI response and display it in the Streamlit app
                answer_list = []
                for input_str in input_data:
                    prompt_str = prompt + input_str
                    answer = generate_answer(prompt_str)
                    answer_list.append(answer)
                st.write("OpenAI's answer:")
                st.write(answer_list)

    else:
        st.write("Sorry for the Incovinience the data is not available")
with st.expander("1.CURRENT WEATHER"):
    pass
        
    
    

def get_text():
    pass

    



