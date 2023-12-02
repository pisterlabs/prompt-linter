from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 


doubt_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Solve doubt for kids on {topic}'
)

# Llms
llm = OpenAI(temperature=0.9) 


doubt_chain = LLMChain(llm=llm, prompt=doubt_template, verbose=True, output_key='doubt')


# Show stuff to the screen if there's a prompt
def doubt(prompt):
    doubt = doubt_chain.run(prompt)
    return doubt
    