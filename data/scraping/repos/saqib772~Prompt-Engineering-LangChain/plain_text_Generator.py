import os

os.environ['OPENAI_API_KEY'] = '###'

## ## Plain Conditional Generation
from langchain.llms import OpenAI
llm = OpenAI(model_name='text-davinci-003', 
             temperature=0.9, 
             max_tokens = 256)

text = "Why did the Truck cross the Sky?"

print(llm(text))

#Basic Template
from langchain import PromptTemplate


universe_template = """
I want you to act as a Universe assistant for new User Questions.

Return a list of Answers. Each Answer should be short, catchy and easy to remember. It shoud relate to the type of Question you are Answering.

What are some Concise Answers for a Question about {Question_desription}?
"""

prompt = PromptTemplate(
    input_variables=["Question_desription"],
    template=universe_template,
)

# An example prompt with one input variable
prompt_template = PromptTemplate(input_variables=["Question_desription"], template=universe_template)

description = " Can Humans will able to colonize Mars "

## to see what the prompt will be like
prompt_template.format(Question_desription=description)

## querying the model with the prompt template
from langchain.chains import LLMChain


chain = LLMChain(llm=llm, prompt=prompt_template)

# Run the chain only specifying the input variable.
print(chain.run(description))
