# Import necessary modules
import os 
from apikey import apikey 
import streamlit as st 
from langchain.llms import OpenAI
from langchain import PromptTemplate

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey

# Set the title of the Streamlit app
st.title('🦜🔗 Commander Spock\'s Q&A 🖖🏽')

# Create a text input for the user to enter their prompt
user_input = st.text_input('Plug in your prompt here')

# Define a template for generating responses
template = """Answer the question as Spock from the Star Trek TC=V series. You will include Vulcan stories and proverbs. If the
question cannot be answered using the information provided answer with "I do not know".

Question: {query}

Answer: """

# Initialize an OpenAI language model with a temperature of 0.9
llm = OpenAI(temperature=1.4) 

# Generate a response from the language model
if user_input: 
    # Format the template with the user's input
    prompt_string = template.format(query=user_input)
    response = llm(prompt_string)
    st.write(response)