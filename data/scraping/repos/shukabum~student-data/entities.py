import sys
sys.path.append('backendPython')
from llms import *
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv()) # read local .env file
from langchain.chains import LLMChain
from langchain.prompts import  FewShotPromptTemplate, PromptTemplate

# giving few-shot prompts
examples =[

{'Query' : 'name 10 companies which came for cgpa below 7' , 
'Answer' : ['Company' , 'CGPA'] },

{"Query" : "Companies which selected more than 10 students for cgpa below 7" , 
"Answer" : ["Company" , "Selected" , "CGPA"]} ,

{"Query" : "The ctc recieved by students for cgpa below 6",
"Answer" : ["CGPA" , "CTC"]} ,

{"Query" : "The job location for companies which came for cgpa below 7",
"Answer" : [ "Venue" ,"Company" , "CGPA"] },
]

example_formatter_template = """
Query :  {Query}
Answer : {Answer}
"""
example_prompt = PromptTemplate(
    input_variables=['Query','Answer'],
    template=example_formatter_template,
)

prefix = ''' 

You are supposed to extract the list of nodes from the given natural query.
You can refer to  below examples for getting idea of how to do the task.

'''

suffix = '''
You have the following nodes available in the database :
--> Company       --> Selected       --> CTC           
--> CGPA          --> JobProfile    --> Venue     

Stick to the format instructions, no dictionary allowed, only list of nodes allowed.


Query :
{natural_query}


Do not put back-ticks(`) in the output.
 '''


few_shot_prompt = FewShotPromptTemplate( 
    examples=examples,
    # prompt template used to format each individual example
    example_prompt=example_prompt,

    # prompt template string to put before the examples, assigning roles and rules.
    prefix=prefix ,
    
    # prompt template string to put after the examples.
    suffix=suffix ,
    
    # input variable to use in the suffix template
    input_variables=[ "natural_query"],
    example_separator="\n", 
)

get_nodes_chain = LLMChain(llm=llm, prompt=few_shot_prompt, verbose=False,)

# print(get_nodes_chain.run(natural_query="name 10 companies which came for cgpa below 7 and ctc above 20"))


