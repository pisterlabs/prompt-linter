import os
import openai
import streamlit as st

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file
_ = load_dotenv(find_dotenv())

# Set the OpenAI API key from the environment variable
openai.api_key = os.environ['OPENAI_API_KEY']

def main():
    
     # Set the title of the Streamlit application
    st.title("Email GPT Creator") 
    
     # Prompt the user to input their name, topic, reasons, and recipients for the email
    name = st.text_input("Enter your name:")
    topic = st.text_input("Enter topic (leave, wfh, etc):")
    reasons = st.text_input("State some reasons (sickness, personal work, etc):")
    recipeints = st.text_input("Who is the email for?:")
    
    # Define the email template using PromptTemplate from the langchain module
    email_template = PromptTemplate(
        input_variables = ["topic", "reasons", "recipients", "name"],
        
        template = """Write a professional email to {recipients} about {topic} for the following reasons {reasons} \ 
        Make sure the reasons are clearly stated and the email is professional. \
        Be to the point and concise. \
        Be sure to put 'thanks and regards' and mention {name} in the next line below greeting. \
            """
    )
    
      # Create an instance of the OpenAI language model
    llm = OpenAI(temperature=0.5)
    
    # Create a chain using the language model and the email template
    email_chain = LLMChain(llm=llm, prompt=email_template)
    
    # Check if all the required inputs and the OpenAI API key are provided
    if name and topic and reasons and recipeints and openai.api_key:
        
        # Print the inputs for debugging purposes
        print("name", name)
        print("topic", topic)
        print("reasons", reasons)   
        print("recipeints", recipeints)
        
         # Generate the email using the email chain and the provided inputs
        response = email_chain.run(topic=topic, reasons=reasons, recipients=recipeints, name=name)
        
        # Display the generated email using Streamlit
        st.write(response) 
    
if __name__ == "__main__":  
    main()