from dotenv import load_dotenv
import re

load_dotenv()

# variable_pattern = r"\{([^}]+)\}"
# input_variables = re.findall(variable_pattern, template)

from langchain import FewShotPromptTemplate, PromptTemplate, LLMChain
from langchain.llms import OpenAI

llm = OpenAI(model="text-davinci-003", temperature=0)

examples = [
    {"animal": "lion", "habitat": "savanna"},
    {"animal": "polar bear", "habitat": "Arctic ice"},
    {"animal": "elephant", "habitat": "African grasslands"},
]

example_template = """
Animal: {animal}
Habitat: {habitat}
"""

example_variables = re.findall(r"\{([^}]+)\}", example_template)
example_prompt = PromptTemplate(
    template=example_template, input_variables=example_variables
)

dynamic_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Identify the habitat of the given animal",
    suffix="Animal: {input}\nHabitat: ",
    input_variables=["input"],
    example_separator="\n\n",
)

chain = LLMChain(llm=llm, prompt=dynamic_prompt)

input_data = {"input": "tiger"}
response = chain.run(input_data)

print(f"Question: {input_data['input']}")
print(f"AI Response: {response}\n")

dynamic_prompt.save("awesome_prompt.json")