# Testing multiple api calls to gpt and serpapi
from serpapi import GoogleSearch
import os
import ast
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


# API Keys
os.environ['SERPAPI_API_KEY'] = 'SERPAPI_API_KEY'

#NOTE TO SELF.  Past this command in the CLI to display UI: streamlit run twotimes.py [ARGUMENTS]

#app framework with text input and title
st.title('DIY Planning with AI')
topic = st.text_input("First, let's start with the reason for your visit? ")


# Prompt Templates
# Acknowledge the user's reason for visiting.  Let them know you will be helping them with a list of tools and supplies.
acknowledge_template = """You are a DIY guide.  You will be helping this person create a list of tools and supplies needed to complete a project. 
You will also be helping them with a list of steps to complete the project.  For now, just acknowledge their reason for visiting based on their topic
Topic: {topic} Acknowledgement:"""

prompt = PromptTemplate(
    input_variables = ['topic'],
    template=acknowledge_template
)

formatted_prompt = prompt.format(topic=topic)

# Prompt Templates - List of tools and supplies
list = PromptTemplate(
    input_variables = ['topic'],
    template="""Provide a short list of primary tools and supplies needed to complete the project. This will be a data response only containing 
    array entries, so please don't include bullet points, dashes, titles or subtitles. Project: {topic}
    """
)

list_prompt = list.format(topic=topic)

# Acknowledge the user's reason for visiting.  Let them know you will be helping them with a list of tools and supplies.
llm = OpenAI( temperature=0, openai_api_key='OPENAI_API_KEY')
st.write(llm.predict(formatted_prompt))

# Save an array of tools and supplies to a variable
list_of_tools1 = []
llm2 = OpenAI( temperature=0, openai_api_key='OPENAI_API_KEY')
list_of_tools = llm2.predict(list_prompt)

# Convert the string to an array
arr = ast.literal_eval(list_of_tools)

st.write("")
st.write("You can get them Home Depot.  Here are the prices and links to each:")

#set an array variable that will hold the sum of the array product values
total = 0

# Create a loop that will iterate through the array and call the serpapi api for each item in the array
for i in range(len(arr)):
    st.write(arr[i])
    params = {
    "engine": "home_depot",
    "q": arr[i], # Search each item in the array
    "api_key": "SERPAPI_API_KEY"
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    products = results["products"][0:1]

    for product in products:
        print(product["title"])
        st.write(product["title"])
        print(product["price"])
        st.write(product["price"])
        print(product["link"] + "\n")
        st.write(product["link"] + "\n")
        total = total + product["price"]


# Print the grand total of the project
st.write("")
st.write("Grand total for all the supplies and parts will cost: " + str(total))
