from langchain import PromptTemplate

location_template = """
I live in {city}, {state}.
"""

prompt = PromptTemplate.from_template(template=location_template)

print(prompt.input_variables)

formatted_prompt = prompt.format(city="Calgary", state="Alberta")

print(formatted_prompt)