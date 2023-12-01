from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import streamlit as st 
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
llm = OpenAI(model_name="text-davinci-003")

def generate_sql_query(user_input):
    template = """
    You are a helpful assistant that can generate SQL queries based on user input.

    User Input: {user_input}
    """
    script_template = PromptTemplate(
    input_variables = ['user_input'], 
    template=template
    )

    script_memory = ConversationBufferMemory(input_key='user_input', memory_key='chat_history')
    
    # LLM
    sql_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

    generated_query = sql_chain.run(user_input)
    return generated_query

# App framework
st.title('SQL Query Generator')
user_input = st.text_area('Code Snippet or Function')

if st.button('Generate Query'):
    sql_query = generate_sql_query(user_input)
    st.write(sql_query)
    print(sql_query)


