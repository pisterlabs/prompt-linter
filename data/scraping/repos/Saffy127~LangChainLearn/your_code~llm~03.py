from langchain import PromptTemplate

complex_template="""
{name} is {age} years old and enjoys {hobby}.
"""

prompt = PromptTemplate(
 input_variables=["name", "age", "hobby"],
  template=complex_template,
)

formatted_prompt = prompt.format(name="Isaac", age="22", hobby="coding")

print(formatted_prompt)
