from dotenv import load_dotenv
import re

load_dotenv()

# variable_pattern = r"\{([^}]+)\}"
# input_variables = re.findall(variable_pattern, template)

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

llm = OpenAI(model="text-davinci-003", temperature=0)

template = """
Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don't know".
Context: Quantum computing is an emerging field that leverages quantum mechanics to solve complex problms faster than classical computing.
###
Question: {query}
Answer: 
"""

template_variables = re.findall(r"\{([^}]+)\}", template)
prompt_template = PromptTemplate(template=template, input_variables=template_variables)

chain = LLMChain(llm=llm, prompt=prompt_template)

input_data = {
    "query": "What is the main advante of quantum computing over classical computing?"
}

response = chain.run(input_data)

print(f"Question: {input_data['query']}")
print(f"AI Response: {response}\n")
