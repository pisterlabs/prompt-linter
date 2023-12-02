import os

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def generate_docstring(code: str):
    # Create an instance of the OpenAI class
    llm = OpenAI(temperature=0, max_tokens=len(code)) # type: ignore

    # Create an instance of the prompt class    
    prompt = PromptTemplate.from_template("I need you to create a docstring for each function and class present in the following code, just return the same code as provided but adding a docstring: {code}")

    # Create an instance of the LLMChain class
    chain = LLMChain(llm=llm, prompt=prompt)

    # Generate the docstring
    output_code = chain.run(code)

    # Return the code with docstring
    return output_code