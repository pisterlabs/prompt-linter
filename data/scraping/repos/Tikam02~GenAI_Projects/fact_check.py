from dotenv import load_dotenv
import os
import streamlit as st
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# load the environment variables from the .env file
load_dotenv()

# Access the environment variables
api_key = os.getenv("OPENAI_API_KEY")


# Set the title
st.title("Fact Checker - With Sequential outputs")


if api_key:
	llm = OpenAI(temperature=1, openapi_api_key=api_key)
else:
	st.warning("Enter your OPENAI API-KEY")


# Add a text input box for the user's question
user_question = st.text_input(
	"Enter Your Question : ",
	placeholder = "NASA sells best icecream in the world",
)


# Generating the final answer to the user's question using all the chains

if st.button("Tell me about it", type="primary"):
	# chain 1: Generating a rephrased version of the user's question
	template = """{question}\n\n"""
	prompt_template = PromptTemplate(input_variables=["question"], template= template)
	question_chain = LLMChain(llm=llm, prompt=prompt_template)

	# chain 2: Generating assumptions made in the statement
	template = """Here is the statement: 
		{statement}
		Make a bullet point list of the assumption you made when producing the statement.\n\n"""
	prompt_template = PromptTemplate(input_variables=["statement"],template=template)
	assumptions_chain = LLMChain(llm=llm, prompt=prompt_template)
	assumptions_chain_seq = SimpleSequentialChain(
	    chains=[question_chain,assumptions_chain],verbose=True
	)
    
	# Chain 3: Fact Checking the assumptions
	template = """Here is a bulltet point of assumptions:
	{assertions}
    For each assertion, determine whether it is true or false.If it is false, explain why.\n\n
    """
	prompt_template = PromptTemplate(input_variables=["assertions"],template=template)
	fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template)
	fact_checker_chain_seq = SimpleSequentialChain(
        chains=[question_chain,assumptions_chain, fact_checker_chain],verbose=True
	)
    
	# FInal Chain: Generating the final answer to the user's question based on the facts and assumptions
	template = """In light of the above facts, how would you answer the question '{}'""".format(
        user_question
	)
	template = """{facts}\n"""+template
	prompt_template = PromptTemplate(input_variables=["facts"],template=template)
	answer_chain = LLMChain(llm=llm, prompt=prompt_template)
	overall_chain = SimpleSequentialChain(
        chains = [question_chain, assumptions_chain,fact_checker_chain,answer_chain],
        verbose=True,
	)

    # Running all the chains on the user's question and displaying the final answer
	st.success(overall_chain.run(user_question))