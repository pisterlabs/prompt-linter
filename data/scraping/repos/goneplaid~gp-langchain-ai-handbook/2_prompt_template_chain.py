from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
import os

davinci = OpenAI(model_name='text-davinci-003')
os.environ['OPENAI_API_TOKEN'] = os.environ['OPENAI_API_KEY']

template = """Question: {question}

Answer: """
prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

llm_chain = LLMChain(
    prompt=prompt,
    llm=davinci
)


qs = [
    {'question': "Which NFL team won the Super Bowl in the 2010 season?"},
    {'question': "If I am 6 ft 4 inches, how tall am I in centimeters?"},
    {'question': "Who was the 12th person on the moon?"},
    {'question': "How many eyes does a blade of grass have?"}
]

print(llm_chain.generate(qs))
