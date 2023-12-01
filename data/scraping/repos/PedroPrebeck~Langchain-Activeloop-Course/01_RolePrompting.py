from dotenv import load_dotenv
import re

load_dotenv()

# variable_pattern = r"\{([^}]+)\}"
# input_variables = re.findall(variable_pattern, template)

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

llm = OpenAI(model="text-davinci-003", temperature=0)

template = """
As a futuristic robot band conductor, I need you to help me come up with a song title.
What's a cool song title for a song about {theme} in the year {year}?
"""

input_variables = re.findall(r"\{([^}]+)\}", template)

prompt = PromptTemplate(input_variables=input_variables, template=template)

user_input_theme = input("What's your favorite futuristic theme? ")
user_input_year = input("What's your favorite futuristic year? ")

input_data = {"theme": user_input_theme, "year": user_input_year}

chain = LLMChain(llm=llm, prompt=prompt)

response = chain.run(input_data)

print(f"Theme: {input_data['theme']}")
print(f"Year: {input_data['year']}")
print(f"AI-generated song title: {response}")
