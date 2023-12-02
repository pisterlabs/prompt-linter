import streamlit as st
from llmConstants import chat
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
from json.decoder import JSONDecodeError

############################################ SPELL CHECKING ######################################################################


#Initialise LLM chain with the prompt
def initialise_LLM_chain(llm, prompt):
    return LLMChain(llm=llm, prompt=prompt)

#Creates a spell checker prompt that helps to check for any grammatical or spelling errors
def create_spell_checker_prompt(): 

	#Spell checker prompt template
	spell_checker_prompt_template = """
	Correct any grammatical, spelling errors in the question below. 
	Output only the corrected version and nothing else
	Question: {question}
	Corrected version: 
	"""
		
	#Create the spell_checking_prompt
	spell_checking_prompt = PromptTemplate(
		template= spell_checker_prompt_template,
		input_variables=["question"],
	)
	return spell_checking_prompt


#Run the spell check
def run_spell_check(query):
	#Create the prompt for spell check
	spell_checker_prompt = create_spell_checker_prompt()
	#Initialise the LLM chain with the prompt for spell check
	spell_checker_llm_chain = initialise_LLM_chain(chat, spell_checker_prompt)
	#Get the response from the LLM
	spell_checker_llm_response = spell_checker_llm_chain(query)
	return spell_checker_llm_response['text']

############################################ RELEVANCY CHECKING ######################################################################

#Creates a relevant question checker prompt that determines whether the user question is relevant or not
def create_relevant_qn_checker_prompt():
	#Set up a prompt template 
	relevant_qn_checker_prompt_template = """
	[INST]<<SYS>>
	Check the question given by the user to see the question is related to finding information related to a topic
	Answer either Relevant or Irrelevant in 1 word and nothing else
	Question: {question}
	Answer: 
	"""
	#Input examples for the llm to check against
	examples = [{'question': 'Is the article relevant to a topic?',
				},
             	{'question': 'Does the article mention a topic?',
				},
                { 'question': 'Regarding topic, what did the article mention?',
				},
                {'question' : 'What is mentioned about topic?',
				}]
		
	#Create a prompt without the examples
	relevant_qn_checker_prompt = PromptTemplate(input_variables=["question"], template= relevant_qn_checker_prompt_template)
      
	#Create a prompt template including the examples
	relevant_qn_checker_few_shot_prompt = FewShotPromptTemplate(
		examples=examples,
		example_prompt=relevant_qn_checker_prompt,
		suffix="Question: {question}",
		input_variables=["question"]
	)
	return relevant_qn_checker_few_shot_prompt


#Runs the relevancy check
def run_relevancy_check(query):
	#Create the prompt for relevancy
	relevant_qn_checker_prompt = create_relevant_qn_checker_prompt()
	#Initialise the LLM chain with the prompt for relevancy
	relevant_qn_checker_llm_chain = initialise_LLM_chain(chat, relevant_qn_checker_prompt)
	#Get the response from the LLM
	relevant_qn_checker_llm_response = relevant_qn_checker_llm_chain(query)
	return relevant_qn_checker_llm_response['text']

#Checks the user input
def process_user_input(query):
	#Run the spell check to get the corrected question
	corrected_question = run_spell_check(query)
	print(corrected_question)
	#After cleaning the grammatical and other errors check whether the question is relevant
	relevant_output = run_relevancy_check(corrected_question)
	print(relevant_output)
	return corrected_question, relevant_output.lower()


	
