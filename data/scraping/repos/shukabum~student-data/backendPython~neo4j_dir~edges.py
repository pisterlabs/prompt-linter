from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
import sys
sys.path.append('backendPython')
from llms import *

import json
# giving few-shot prompts
examples = [
    { 'Text' : 'Company_to_Selected', 
      'Expected' : 'The number of students selected by the company'},
    { 'Text' : 'Company_to_CTC',
      'Expected' : 'The salary offered by the company'},
    { 'Text' : 'CTC_to_Company',
      'Expected' : 'The companies which offered the given salary'},
    { 'Text' : 'Company_to_Venue',
       'Expected' : 'The work location set by the company'},
    { 'Text' : 'Company_to_CGPA',
      'Expected' : 'The minimum cgpa required by the company for the job profile'},
    { 'Text' : 'Company_to_Job Profile',
      'Expected' : 'The job profile offered by the company'
    }, 
]
example_formatter_template = """
'Text' : {Text},
'Expected' : {Expected}\n
"""
example_prompt = PromptTemplate(
    input_variables=["Text", "Expected"],
    template=example_formatter_template,
)

few_shot_prompt = FewShotPromptTemplate( 
    examples = examples,

    # prompt template used to format each individual example
    example_prompt=example_prompt,

    # prompt template string to put before the examples, assigning roles and rules.
    prefix="Here are some examples of how to explain the relationship between 2 nodes :\n\n",
    
    # prompt template string to put after the examples.
    suffix="\n\nNow, given a new relationship, explain the relationship:\n\nText: {input}\nExpected:",
    
    # input variable to use in the suffix template
    input_variables=["input"],
    example_separator="\n", 
)

chain = LLMChain(llm=llm, prompt=few_shot_prompt)

edges_dict = {}

cols = ['Company', 'Selected',  'CTC', 'CGPA', 'JobProfile', 'Venue']

for i in range(6):
  for j in range(6):
    if i==j: 
      continue
    source , target= cols[i] , cols[j]
    type = '{}_to_{}'.format(source, target)
    edges_dict[type] = chain.run(input=type)


json.dump(edges_dict, open('backendPython/neo4j/edges.json', 'w'))