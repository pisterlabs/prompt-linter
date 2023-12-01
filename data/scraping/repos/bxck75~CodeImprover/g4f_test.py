# pip install git+https://github.com/MIDORIBIN/langchain-gpt4free.git

import RoleSelector
from g4f import logging,models
from langchain.llms.base import LLM
from langchain_g4f import G4FLLM
from langchain.chains import SimpleSequentialChain

""" logging=True

Role1 = RoleSelector.select_role()
Role2 = RoleSelector.select_role()
Role3 = RoleSelector.select_role()

print("A {} talking to a {} and a {}".format(Role1[0],Role2[0],Role2[0]))

response1 = []
response2 = []
response3 = []

system_prompt1 = f"You are a {Role1[0]}"
system_prompt2 = f"You are a {Role2[0]} {Role2[1]}"
system_prompt3 = f"You are a {Role3[0]} {Role3[1]}"

user_prompt1 = f"{Role1[1]}, please introduce yourself and tell me your role"
user_prompt2 = f"{Role2[0]}, please introduce yourself and tell me your role"
user_prompt3 = f"{Role3[0]}, please introduce yourself and tell me your role"
"""
from langchain.prompts import (PromptTemplate, 
                               ChatPromptTemplate, 
                               StringPromptTemplate ,
                               ChatMessagePromptTemplate)
from langchain.schema import SystemMessage, HumanMessage,ChatMessage, BasePromptTemplate, format_document
from langchain.chains import LLMChain, SimpleSequentialChain, LLMRouterChain
from langchain.llms.base import LLM
from langchain.prompts.chat import ChatPromptValue


llm1: LLM = G4FLLM(model=models.gpt_35_turbo)
#prompt = StringPromptTemplate(prompt=user_prompt1, role=system_prompt1)

"""
prompt5 = ChatMessagePromptTemplate(input_variables=["role_name","role_description"], 
                        template="You are a {role_name}, {role_description}.")

chain1 = LLMChain(llm=llm1, prompt=prompt)
chain2 = LLMChain(llm=llm1, prompt=prompt)
chain3 = LLMChain(llm=llm1, prompt=prompt)

# Create a SimpleSequentialChain with the three LLMs
sim_seq_chain = SimpleSequentialChain(chains=[chain1,chain2,chain3])

# Start the conversation
while user_input != "exit":
    response = sim_seq_chain("kak")
    print("Assistant:", response)
    user_input = input("User: ")
 """
from langchain.schema import Document
from langchain.prompts import PromptTemplate

with open("/media/k00b404/04ef09de-2d9f-4fc2-8b89-de7dc0155e26/coding_folder_new/CamelGalore/gpt4free/etc/testing/TimeLogger_generated_2_7_improvement_generated_2_7_improvement.py", "r") as file:
    doc_text = file.read()

doc = Document(page_content=doc_text, metadata={"page": "1"})
prompt = PromptTemplate.from_template("Page {page}: {page_content}")
prompt_from_script = format_document(doc, prompt)

#llm1(prompt_from_script)

#chain1 = LLMChain(llm=llm1, prompt=prompt_from_script)

#print(chain1)

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import autopep8


llm1: LLM = G4FLLM(model=models.gpt_35_turbo)
prompt_template = "Tell me a {adjective} joke"
prompt = PromptTemplate(
    input_variables=["adjective"], template=prompt_template
)
llm_chain = LLMChain(llm=llm1, prompt=prompt)

def check_syntax_errors(code: str) -> None:
    try:
        compile(code[0].to_string(), "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax Error: {e}")


def format_code(code: str) -> str:
    return autopep8.fix_code(code)




user_input = "user: improve the script"
while user_input != "exit":
    response = llm_chain(prompt_from_script)
    print("Assistant:", response)
    
    response = llm_chain(prompt_from_script)
    #check syntax
    check_syntax_errors(code=response)
    # Format
    fcode = format_code(response)
    print(fcode)
    user_input = input("User: ")



#[SystemMessage(content='You are a helpful assistant that translates English to French.', additional_kwargs={}),
#    HumanMessage(content='I love programming.', additional_kwargs={})]
#output = response.format(input_language="English", output_language="French", text="I love programming.").to_string()
#ChatPromptValue(messages=[SystemMessage(content='You are a helpful assistant that translates English to French.', additional_kwargs={}), HumanMessage(content='I love programming.', additional_kwargs={})])
# or alternatively
#output_2 = chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.").to_string()

#assert output == output_2

#output = prompt_from_script.format(input_language="English", output_language="French", text="I love programming.")
