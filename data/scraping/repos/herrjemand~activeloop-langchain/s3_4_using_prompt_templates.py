from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain import PromptTemplate, LLMChain, FewShotPromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(model="text-davinci-003", temperature=0)

template = """Answer the question based on the context below.
If the question cannot be answered using the information provided, answer with "I don't know".

Context: Quantum computing is an emerging field that leverages the principles of quantum mechanics to
to solve complex problems faster than classical computers.
...

Question: {question}
Answer:
"""

prompt_template = PromptTemplate(
    template=template,
    input_variables=["question"]
)

chain = LLMChain(llm=llm, prompt=prompt_template)

input = "What is the main advantage of the quantum computing?"

response = chain.run({"question": input})

print("Input:", input)
print("Output:", response)


examples = [
    {"animal": "lion", "habitat": "savanna"},
    {"animal": "polar bear", "habitat": "Arctic ice"},
    {"animal": "elephant", "habitat": "African grasslands"}
]

example_template = """
Animal: {animal}
Habitat: {habitat}
"""

example_prompt = PromptTemplate(
    template=example_template,
    input_variables=["animal", "habitat"]
)

dynamic_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    input_variables=["input"],
    prefix="Identify the habbit of the given animal",
    suffix="Animal: {input}\nHabitat: ",
    example_separator="\n\n",
)


dynamic_prompt.save("dynamic_prompt.json")
chain = LLMChain(llm=llm, prompt=dynamic_prompt)

response = chain.run({"input": "tiger"})

print("Input:", "llama")
print("Output:", response)