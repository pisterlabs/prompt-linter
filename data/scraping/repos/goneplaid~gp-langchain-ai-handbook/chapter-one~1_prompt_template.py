from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
import os

davinci = OpenAI(model_name='text-davinci-003')
os.environ['OPENAI_API_TOKEN'] = os.environ['OPENAI_API_KEY']

template = """Question: {question}

Answer: """
prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

# user question
question = "Which NFL team won the Super Bowl in the 2010 season?"

llm_chain = LLMChain(
    prompt=prompt,
    llm=davinci
)

print(llm_chain.run(question))
